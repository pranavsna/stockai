"""
Since the agent now runs every 20-30 minutes, running the exact same
full S&P 500 scan every time is wasteful and won't surface anything
new intra-day. This module rotates between a few different discovery
strategies across runs, tracked via a counter persisted in state.json.

Modes:
  - "core"          : the original 2-day-horizon scan across the S&P 500.
  - "long_horizon"   : same idea, but the model is trained to predict a
                       ~10-trading-day move instead of 2, with different
                       hyperparameters (fewer, shallower trees - less
                       prone to overfitting on a noisier longer-horizon
                       label). Surfaces different candidates than "core".
  - "news_discovery" : skips the price model's tail of the universe and
                       instead scans general market-news RSS feeds for
                       which S&P 500 companies are getting unusually
                       heavy headline volume right now, then runs the
                       standard model only on that shortlist.

Every run still does position monitoring (see agent.py) regardless of
which discovery mode is selected.
"""

import re
from collections import Counter

import feedparser
import pandas as pd

import stock_model as sm

MODES = ["core", "long_horizon", "news_discovery"]

GENERAL_NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=stock+market+upgrade&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=stock+surges&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=earnings+beat&hl=en-US&gl=US&ceid=US:en",
]

LONG_HORIZON_DAYS = 10


def pick_mode(run_counter):
    return MODES[run_counter % len(MODES)]


# ---------------------------------------------------------------------
# core: same as the original live scan, just factored out here
# ---------------------------------------------------------------------

def scan_core(candidates, prob_threshold=0.95):
    """
    Returns {ticker: (prob, latest_row_df)} for tickers the standard
    2-day model flags. `candidates` should be a dict of {ticker: hist}
    from a batch fetch.
    """
    return _scan(candidates, horizon=2, prob_threshold=prob_threshold,
                 n_estimators=200, max_depth=6, learning_rate=0.1)


# ---------------------------------------------------------------------
# long_horizon: different label window + lighter hyperparameters
# ---------------------------------------------------------------------

def scan_long_horizon(candidates, prob_threshold=0.90):
    """
    Trains on a longer forward-return label (~10 trading days) with
    fewer/shallower trees, since a longer, noisier label tends to
    overfit faster with the original's deeper/heavier config. Slightly
    lower probability threshold since the label distribution differs
    and 0.95 on a differently-calibrated model isn't the same bar.
    """
    return _scan(candidates, horizon=LONG_HORIZON_DAYS, prob_threshold=prob_threshold,
                 n_estimators=120, max_depth=4, learning_rate=0.08)


def _scan(candidates, horizon, prob_threshold, n_estimators, max_depth, learning_rate):
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    flagged = {}

    for ticker, hist in candidates.items():
        try:
            data, predictors = sm.compute_features_from_hist(hist, horizon=horizon)
            if data.empty or len(data) < 40:
                continue

            train_data = data.iloc[:-1]
            latest_row = data.iloc[-1:]
            if len(train_data) < 30:
                continue

            X = train_data[predictors]
            y = train_data["Target"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pos = (y == 1).sum()
            neg = (y == 0).sum()
            scale_pos_weight = neg / pos if pos > 0 else 1

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                verbosity=0,
            )
            model.fit(X_scaled, y)

            prob = model.predict_proba(latest_row[predictors])[:, 1][0]
            if prob > prob_threshold:
                flagged[ticker] = (float(prob), latest_row)

        except Exception as e:
            print(f"[{ticker}] scan failed: {e}")
            continue

    return flagged


# ---------------------------------------------------------------------
# news_discovery: find tickers with unusually heavy headline chatter,
# then run the standard core model on just that shortlist
# ---------------------------------------------------------------------

def _extract_ticker_mentions(headline, known_tickers_set):
    """
    Crude but cheap: looks for uppercase word-boundary tokens in the
    headline that match a known S&P 500 ticker. Misses tickers only
    referenced by company name, but avoids needing a paid NER/entity
    service - a free ticker-symbol match is good enough as a chatter
    signal, not a precise one.
    """
    tokens = re.findall(r"\b[A-Z]{1,5}\b", headline)
    return [t for t in tokens if t in known_tickers_set]


def find_chatter_tickers(known_tickers, top_n=15, max_headlines_per_feed=40):
    known_set = set(known_tickers)
    mention_counts = Counter()

    for feed_url in GENERAL_NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_headlines_per_feed]:
                title = getattr(entry, "title", "")
                for ticker in _extract_ticker_mentions(title, known_set):
                    mention_counts[ticker] += 1
        except Exception as e:
            print(f"[news_discovery] feed failed ({feed_url}): {e}")
            continue

    return [t for t, _ in mention_counts.most_common(top_n)]


def scan_news_discovery(known_tickers, prob_threshold=0.90):
    chatter_tickers = find_chatter_tickers(known_tickers)
    print(f"[news_discovery] chatter tickers: {chatter_tickers}")

    if not chatter_tickers:
        return {}

    start_date = pd.Timestamp.today() - pd.Timedelta(days=365 * 5)
    end_date_exclusive = pd.Timestamp.today() + pd.Timedelta(days=1)
    hist_by_ticker = sm.batch_fetch_history(chatter_tickers, start_date, end_date_exclusive)

    return scan_core(hist_by_ticker, prob_threshold=prob_threshold)

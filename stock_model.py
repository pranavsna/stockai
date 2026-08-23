import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from multiprocessing import Pool, cpu_count, Manager
import bs4
import requests
import time

def get_stock_tickers():
    """
    Retrieves the list of S&P 500 tickers.

    NOTE (known, intentionally not fixed here): this always
    returns *today's* constituents, even when used inside
    backtest() for historical dates. That introduces
    survivorship bias in the backtest, since it can't account
    for tickers that were added/removed from the index during
    the backtest window. Fixing this properly requires a
    historical-constituents data source (most are paid), so
    it's left as a hook below rather than solved.
    """
    with open('sp500.txt') as file:
        tickers = file.read().split(' ')

    if tickers == ['']:
        response = requests.get(
            'https://stockanalysis.com/list/sp-500-stocks/'
        )
        soup = bs4.BeautifulSoup(
            response.text,
            'html.parser'
        )
        tickers = [
            td.text
            for td in soup.find_all(
                'td',
                class_='sym svelte-1ro3niy'
            )
        ]

        with open('sp500.txt', 'w') as file:
            file.write(' '.join(tickers))

    return tickers


def get_historical_constituents(date, current_tickers):
    return current_tickers


# =========================================================
# FEATURE ENGINEERING (shared by live + backtest paths)
# =========================================================

def compute_features_from_hist(hist):
    """
    Takes a raw OHLCV history DataFrame (as returned by yfinance)
    and returns (data, full_predictors), or (empty df, None) if
    there isn't enough data to build features.

    This is factored out so both the live path (fetch_and_prepare_data)
    and the backtest path (which batch-downloads history for many
    tickers at once) share exactly one feature-engineering
    implementation instead of two copies that can drift apart.
    """
    if hist is None or len(hist) == 0:
        return pd.DataFrame(), None

    data = hist[["Close"]].rename(
        columns={'Close': 'Actual_Close'}
    )

    # Predicting 2-day forward price movement.
    #
    # FIX: hist["Close"].shift(-2) is NaN for the last 2 rows.
    # `NaN > x` evaluates to False in pandas, so casting straight
    # to int previously turned "we don't know yet" into a hard,
    # fabricated "Decrease" (0) label for the 2 most recent rows.
    # Those rows are also the ones closest to whatever date you're
    # predicting on, so they got trained on with a wrong label.
    # Explicitly mask them as NaN instead, and let dropna() remove
    # them like every other incomplete row.
    future_close = hist["Close"].shift(-2)
    target = np.where(
        future_close.isna(),
        np.nan,
        (future_close > hist["Close"]).astype(float)
    )
    data["Target"] = target

    prev_data = hist.shift(1)

    predictors = [
        "Close",
        "Volume",
        "Open",
        "High",
        "Low"
    ]

    data = data.join(
        prev_data[predictors]
    )

    data = data.dropna(subset=predictors + ["Target"])
    data["Target"] = data["Target"].astype(int)

    # Various features for the model to consider
    data["weekly_mean"] = (
        data["Close"].rolling(7).mean()
        / data["Close"]
    )

    data["quarterly_mean"] = (
        data["Close"].rolling(90).mean()
        / data["Close"]
    )

    data["weekly_trend"] = (
        data["Target"]
        .shift(1)
        .rolling(7)
        .sum()
    )

    data["open_close_ratio"] = (
        data["Open"] / data["Close"]
    )

    data["high_close_ratio"] = (
        data["High"] / data["Close"]
    )

    data["low_close_ratio"] = (
        data["Low"] / data["Close"]
    )

    data["volatility"] = (
        (data["High"] - data["Low"])
        / data["Close"]
    )

    data["daily_return"] = (
        data["Close"].pct_change()
    )

    data["SMA20"] = (
        data["Close"].rolling(20).mean()
    )

    data["SMA50"] = (
        data["Close"].rolling(50).mean()
    )

    data["EMA20"] = data["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    data["EMA50"] = data["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    data["MACD"] = (
        data["Close"].ewm(
            span=12,
            adjust=False
        ).mean()
        -
        data["Close"].ewm(
            span=26,
            adjust=False
        ).mean()
    )

    gains = data["Close"].diff().clip(lower=0).rolling(14).mean()
    losses = data["Close"].diff().clip(upper=0).abs().rolling(14).mean()

    # FIX: losses can be exactly 0 (e.g. 14 straight up days), which
    # made RSI blow up to +/-inf and survive the final dropna() since
    # inf isn't NaN. Replace inf explicitly before dropping.
    data["RSI"] = 100 - (100 / (1 + (gains / losses)))
    data["RSI"] = data["RSI"].replace([np.inf, -np.inf], np.nan)

    data = data.dropna()

    full_predictors = predictors + [
        "weekly_mean",
        "quarterly_mean",
        "weekly_trend",
        "open_close_ratio",
        "high_close_ratio",
        "low_close_ratio",
        "volatility",
        "daily_return",
        "SMA20",
        "SMA50",
        "EMA20",
        "EMA50",
        "MACD",
        "RSI"
    ]

    return data, full_predictors


def fetch_and_prepare_data(ticker):
    hist = yf.Ticker(ticker).history(
        period="5y",
        interval="1d"
    )
    return compute_features_from_hist(hist)


# =========================================================
# MODEL
# =========================================================

def train_and_evaluate_model(
    data,
    full_predictors
):

    X = data[full_predictors]
    y = data["Target"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbosity=0
    )

    model.fit(
        X_scaled,
        y
    )

    return model

def predict_tomorrow(
    data,
    model,
    predictors
):

    latest_data = data.iloc[-1:].copy()

    preds = model.predict_proba(
        latest_data[predictors]
    )[:, 1]

    prediction = (
        1 if preds[0] > 0.95 else 0
    )

    return prediction


# =========================================================
# BATCH HISTORY FETCH (reused by the agent for live data too)
# =========================================================

def batch_fetch_history(tickers, start_date, end_date_exclusive):
    """
    Fetches history for many tickers in ONE request instead of one
    yf.Ticker(...).history() call per ticker.
    """
    raw = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date_exclusive.strftime("%Y-%m-%d"),
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )

    out = {}

    for t in tickers:
        try:
            sub = raw[t].dropna(how="all")
            if not sub.empty:
                out[t] = sub
        except (KeyError, IndexError):
            continue

    return out


def get_spy_return(start_date, end_date):
    """Buy-and-hold SPY return over the same window, for comparison."""
    hist = yf.Ticker("SPY").history(
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
    )
    if hist.empty:
        return None
    buy_price = hist.iloc[0]["Close"]
    sell_price = hist.iloc[-1]["Close"]
    return (sell_price / buy_price) - 1

"""
Daily agent run.

For each candidate ticker (i.e. not already held within its 2-day
hold window):
  1. Fetch price history, compute features (existing pipeline).
  2. Train the model on all rows EXCEPT the most recent one, then
     score that held-out most-recent row. This is the fix from
     the earlier leak discussion: the row being scored must never
     appear in the training set.
  3. Fetch recent news headlines and score sentiment (free, VADER).
  4. Buy only if the model says "increase" AND sentiment isn't
     clearly negative (a simple veto filter, not a second vote).

Tickers already held get checked against their sell_date and sold
(logically - this only recommends, it does not place real trades)
once the 2-trading-day hold is up.

Run with `python agent.py`. Designed to be invoked on a schedule
by the GitHub Actions workflow in .github/workflows/agent.yml.
"""

import argparse
from datetime import date

import pandas as pd

import stock_model as sm
import news_sentiment as ns
import position_tracker as pt
from notifier import send_telegram_message, format_run_summary

MODEL_PROB_THRESHOLD = 0.95   # same threshold as the original script
SENTIMENT_VETO_THRESHOLD = -0.15  # below this, skip the buy even if model says yes
HOLD_TRADING_DAYS = 2
MAX_UNIVERSE = None  # set to an int (e.g. 100) to limit tickers for a faster/cheaper run


def score_ticker(ticker, hist):
    """
    Returns (prediction, prob, latest_close) or None if there's not
    enough data. Trains on all rows except the most recent one, then
    scores that held-out row - this is the leak fix.
    """
    data, predictors = sm.compute_features_from_hist(hist)
    if data.empty or len(data) < 30:
        return None

    train_data = data.iloc[:-1]     # everything except the row being scored
    latest_row = data.iloc[-1:]

    if len(train_data) < 20:
        return None

    model = sm.train_and_evaluate_model(train_data, predictors)
    prob = model.predict_proba(latest_row[predictors])[:, 1][0]
    prediction = 1 if prob > MODEL_PROB_THRESHOLD else 0

    latest_close = float(latest_row["Close"].iloc[0])

    return prediction, float(prob), latest_close


def run(state_path="state.json", dry_run=False):
    today = date.today()
    print(f"=== Agent run: {today} ===")

    state = pt.load_state(state_path)
    still_held, ready_to_sell = pt.get_active_holdings(state, today)

    print(f"Currently held (within hold window): {still_held}")
    print(f"Ready to sell today: {ready_to_sell}")

    # --- Handle sells for positions whose hold period is up ---
    sells = []
    if ready_to_sell:
        sell_hist = sm.batch_fetch_history(
            ready_to_sell,
            pd.Timestamp(today) - pd.Timedelta(days=5),
            pd.Timestamp(today) + pd.Timedelta(days=1),
        )
        for ticker in ready_to_sell:
            info = state[ticker]
            hist = sell_hist.get(ticker)
            if hist is None or hist.empty:
                print(f"{ticker}: no fresh price data to compute sell, leaving in state")
                continue
            sell_price = float(hist["Close"].iloc[-1])
            buy_price = info["buy_price"]
            sells.append({
                "ticker": ticker,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "pct_return": (sell_price / buy_price) - 1,
            })
            pt.remove_position(state, ticker)

    # --- Build candidate universe (exclude anything still held) ---
    all_tickers = sm.get_stock_tickers()
    candidates = [t for t in all_tickers if t not in still_held and t not in ready_to_sell]
    if MAX_UNIVERSE:
        candidates = candidates[:MAX_UNIVERSE]

    print(f"Scoring {len(candidates)} candidate tickers...")

    # Batch price fetch (one request for everything, not one per ticker)
    start_date = pd.Timestamp(today) - pd.Timedelta(days=365 * 5)
    end_date_exclusive = pd.Timestamp(today) + pd.Timedelta(days=1)
    hist_by_ticker = sm.batch_fetch_history(candidates, start_date, end_date_exclusive)

    # Pre-filter to tickers the model likes, THEN fetch news only for
    # those - this avoids doing ~500 RSS fetches every run.
    model_buys = {}
    for ticker, hist in hist_by_ticker.items():
        try:
            result = score_ticker(ticker, hist)
        except Exception as e:
            print(f"{ticker}: scoring failed ({e})")
            continue
        if result is None:
            continue
        prediction, prob, latest_close = result
        if prediction == 1:
            model_buys[ticker] = (prob, latest_close)

    print(f"Model flagged {len(model_buys)} tickers: {list(model_buys.keys())}")

    # --- News sentiment veto, only for model-flagged tickers ---
    sentiments = ns.get_sentiment_batch(list(model_buys.keys()))

    buys = []
    for ticker, (prob, latest_close) in model_buys.items():
        sent_info = sentiments.get(ticker, {})
        mean_compound = sent_info.get("mean_compound")

        if mean_compound is not None and mean_compound < SENTIMENT_VETO_THRESHOLD:
            print(f"{ticker}: model said buy but sentiment vetoed it "
                  f"({mean_compound:.2f})")
            continue

        buys.append({
            "ticker": ticker,
            "model_prob": prob,
            "sentiment": mean_compound,
            "price": latest_close,
        })

        if not dry_run:
            pt.add_position(
                state, ticker, today,
                buy_price=latest_close,
                model_prob=prob,
                sentiment=mean_compound,
                hold_trading_days=HOLD_TRADING_DAYS,
            )

    if not dry_run:
        pt.save_state(state, state_path)

    summary = format_run_summary(
        buys=buys,
        sells=sells,
        skipped_count=len(still_held),
        run_date=today.strftime("%Y-%m-%d"),
    )
    print(summary)
    send_telegram_message(summary)

    return buys, sells


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="state.json")
    parser.add_argument("--dry-run", action="store_true",
                         help="Score and notify but don't write state.json")
    args = parser.parse_args()

    run(state_path=args.state, dry_run=args.dry_run)

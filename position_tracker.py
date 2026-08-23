"""
Tracks which tickers the agent currently considers "held" and
enforces the minimum hold period between runs.

State is a single JSON file, e.g.:

{
  "AAPL": {
    "buy_date": "2026-08-20",
    "sell_date": "2026-08-24",
    "buy_price": 227.31,
    "model_prob": 0.962,
    "sentiment": 0.41
  }
}

This file is meant to be committed back into the repo by the
GitHub Actions workflow after each run, so state persists across
runs for free (no database needed).
"""

import json
import os
from datetime import datetime, timedelta

DEFAULT_STATE_PATH = "state.json"


def load_state(path=DEFAULT_STATE_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_state(state, path=DEFAULT_STATE_PATH):
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def next_trading_day_estimate(from_date, trading_days_ahead):
    """
    Rough calendar-day estimate of N trading days ahead, skipping
    weekends. Doesn't account for market holidays - close enough
    for a 2-day hold window, but note the imprecision if you need
    exact NYSE trading-day accounting.
    """
    d = from_date
    added = 0
    while added < trading_days_ahead:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            added += 1
    return d


def get_active_holdings(state, today):
    """
    Returns (still_held, ready_to_sell):
      still_held    - tickers whose sell_date is still in the future
      ready_to_sell - tickers whose sell_date has arrived/passed
    """
    still_held = []
    ready_to_sell = []

    for ticker, info in state.items():
        sell_date = datetime.strptime(info["sell_date"], "%Y-%m-%d").date()
        if sell_date <= today:
            ready_to_sell.append(ticker)
        else:
            still_held.append(ticker)

    return still_held, ready_to_sell


def add_position(state, ticker, today, buy_price, model_prob, sentiment,
                  hold_trading_days=2):
    sell_date = next_trading_day_estimate(today, hold_trading_days)
    state[ticker] = {
        "buy_date": today.strftime("%Y-%m-%d"),
        "sell_date": sell_date.strftime("%Y-%m-%d"),
        "buy_price": round(float(buy_price), 4),
        "model_prob": round(float(model_prob), 4),
        "sentiment": None if sentiment is None else round(float(sentiment), 4),
    }
    return state


def remove_position(state, ticker):
    state.pop(ticker, None)
    return state

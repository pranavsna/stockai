"""
Alpaca is the source of truth for WHAT is actually held and its
live P/L. This module stores the metadata Alpaca doesn't track per
position (entry date, earliest allowed sell date, why we bought it,
which discovery mode found it) plus a queue of buys that were decided
on while the market was closed and haven't been submitted yet.

State shape (state.json):
{
  "positions": {
    "AAPL": {
      "entry_date": "2026-08-20",
      "min_sell_date": "2026-08-24",
      "rationale": "...",
      "mode": "core"
    }
  },
  "deferred_orders": [
    {
      "ticker": "MSFT",
      "dollar_amount": 850.0,
      "rationale": "...",
      "mode": "long_horizon",
      "queued_at": "2026-08-24T21:10:00",
      "target_time": "2026-08-25T13:30:00"
    }
  ],
  "meta": {
    "run_counter": 7
  }
}
"""

import json
import os
from datetime import datetime, timedelta

DEFAULT_STATE_PATH = "state.json"


def _empty_state():
    return {"positions": {}, "deferred_orders": [], "meta": {"run_counter": 0}}


def load_state(path=DEFAULT_STATE_PATH):
    if not os.path.exists(path):
        return _empty_state()
    with open(path, "r") as f:
        try:
            state = json.load(f)
        except json.JSONDecodeError:
            return _empty_state()
    # Backfill in case of an older/partial state file
    state.setdefault("positions", {})
    state.setdefault("deferred_orders", [])
    state.setdefault("meta", {"run_counter": 0})
    state["meta"].setdefault("run_counter", 0)
    return state


def save_state(state, path=DEFAULT_STATE_PATH):
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def next_trading_day_estimate(from_date, trading_days_ahead):
    """
    Calendar-day estimate of N trading days ahead, skipping weekends.
    Doesn't account for market holidays - fine for a minimum-hold
    check, imprecise right around holidays.
    """
    d = from_date
    added = 0
    while added < trading_days_ahead:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def record_new_position(state, ticker, today, rationale, mode, min_hold_trading_days=2):
    state["positions"][ticker] = {
        "entry_date": today.strftime("%Y-%m-%d"),
        "min_sell_date": next_trading_day_estimate(today, min_hold_trading_days).strftime("%Y-%m-%d"),
        "rationale": rationale,
        "mode": mode,
    }
    return state


def remove_position(state, ticker):
    state["positions"].pop(ticker, None)
    return state


def is_sellable(state, ticker, today):
    """True if this position has cleared its minimum hold period."""
    info = state["positions"].get(ticker)
    if not info:
        return False
    min_sell_date = datetime.strptime(info["min_sell_date"], "%Y-%m-%d").date()
    return today >= min_sell_date


def queue_deferred_order(state, ticker, dollar_amount, rationale, mode, now, next_open):
    for i in state["deferred_orders"]:
      if i["ticker"] == ticker:
        return state
    state["deferred_orders"].append({
        "ticker": ticker,
        "dollar_amount": round(dollar_amount, 2),
        "rationale": rationale,
        "mode": mode,
        "queued_at": now.isoformat(),
        "target_time": next_open.isoformat() if next_open else None,
    })
    return state


def pop_deferred_orders(state):
    """Removes and returns all queued deferred orders."""
    orders = state["deferred_orders"]
    state["deferred_orders"] = []
    return orders


def next_mode_and_increment(state, modes):
    counter = state["meta"]["run_counter"]
    mode = modes[counter % len(modes)]
    state["meta"]["run_counter"] = counter + 1
    return mode

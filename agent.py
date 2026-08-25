"""
Main agent run - designed to run every 20-30 minutes via GitHub
Actions during market hours.

Each run always does two things:
  1. POSITION MONITORING - checks every currently-held Alpaca paper
     position against exit rules (stop-loss / take-profit). Exits
     only fire once the 2-trading-day minimum hold has passed;
     otherwise they're reported as "flagged, waiting on hold".
  2. DISCOVERY - rotates between three strategies (see discovery.py)
     so repeated runs aren't just re-running an identical scan:
       core          -> standard 2-day-horizon model across the S&P 500
       long_horizon   -> same idea, ~10-day horizon, lighter hyperparams
       news_discovery -> shortlist from general market-news chatter,
                          then scores that shortlist with the core model

New buys are sized as a fraction of current account equity, weighted
by a composite score (model probability + sentiment + technicals -
see reasoning.py), capped per-position and by a max concurrent
position count.

If the market is closed when a buy decision is made, the order is
queued in state.json as a deferred order instead of submitted, and
executed automatically on a later run once the market opens.
"""

import argparse
from datetime import date, datetime

import pandas as pd

import stock_model as sm
import news_sentiment as ns
import position_tracker as pt
import discovery
import reasoning
from alpaca_trader import AlpacaTrader
from notifier import send_telegram_message, format_run_summary

MAX_POSITIONS = 8
MAX_ALLOCATION_FRACTION = 0.20     # cap per position as a fraction of equity
MIN_ORDER_DOLLARS = 250
CASH_BUFFER_FRACTION = 0.10        # keep 10% of cash uncommitted

STOP_LOSS_PCT = -0.08
TAKE_PROFIT_PCT = 0.15

MIN_HOLD_TRADING_DAYS = 2
MAX_UNIVERSE = None  # set an int to limit the scan universe for a faster/cheaper run


# =========================================================
# POSITION MONITORING
# =========================================================

def check_exit_rule(unrealized_plpc):
    if unrealized_plpc <= STOP_LOSS_PCT:
        return f"stop-loss hit ({unrealized_plpc*100:+.2f}%)"
    if unrealized_plpc >= TAKE_PROFIT_PCT:
        return f"take-profit hit ({unrealized_plpc*100:+.2f}%)"
    return None


def monitor_positions(trader, state, today, market_open):
    """
    Returns (held_positions_for_display, sells, exit_flagged_waiting).
    Executes real sell orders (via Alpaca) when an exit rule fires and
    the minimum hold has cleared and the market is open.
    """
    live_positions = trader.get_positions()

    held_for_display = []
    sells = []
    exit_flagged_waiting = []

    for ticker, pos in live_positions.items():
        meta = state["positions"].get(ticker)
        if meta is None:
            # Position exists in Alpaca but we have no metadata for it
            # (e.g. manually opened, or state.json was reset). Track it
            # from today with the default hold so it isn't ignored.
            state = pt.record_new_position(
                state, ticker, today,
                rationale="No prior metadata found; tracking from today.",
                mode="unknown",
                min_hold_trading_days=MIN_HOLD_TRADING_DAYS,
            )
            meta = state["positions"][ticker]

        exit_reason = check_exit_rule(pos["unrealized_plpc"])
        sellable = pt.is_sellable(state, ticker, today)

        if exit_reason and sellable and market_open:
            trader.sell_all(ticker)
            sells.append({
                "ticker": ticker,
                "reason": exit_reason,
                "unrealized_plpc": pos["unrealized_plpc"],
                "market_value": pos["market_value"],
            })
            pt.remove_position(state, ticker)
            continue

        if exit_reason and not sellable:
            exit_flagged_waiting.append({
                "ticker": ticker,
                "reason": exit_reason,
                "min_sell_date": meta["min_sell_date"],
            })

        held_for_display.append({
            "ticker": ticker,
            "entry_date": meta["entry_date"],
            "min_sell_date": meta["min_sell_date"],
            "sellable": sellable,
            "unrealized_plpc": pos["unrealized_plpc"],
            "market_value": pos["market_value"],
            "rationale": meta.get("rationale", ""),
        })

    return held_for_display, sells, exit_flagged_waiting


# =========================================================
# DEFERRED ORDER EXECUTION
# =========================================================

def execute_deferred_orders(trader, state, today, mode_label="deferred"):
    """
    If the market is open, submits any orders that were queued while
    it was closed. Returns the list of newly-executed buys (same
    shape as fresh buys) for the notification.
    """
    executed = []
    pending = state["deferred_orders"]
    if not pending:
        return executed

    still_pending = []
    for order in pending:
        try:
            trader.buy_notional(order["ticker"], order["dollar_amount"])
            state = pt.record_new_position(
                state, order["ticker"], today,
                rationale=order["rationale"] + " (executed from deferred queue)",
                mode=order.get("mode", mode_label),
                min_hold_trading_days=MIN_HOLD_TRADING_DAYS,
            )
            executed.append({
                "ticker": order["ticker"],
                "dollar_amount": order["dollar_amount"],
                "score": None,
                "rationale": order["rationale"] + " (was deferred)",
            })
        except Exception as e:
            print(f"[deferred] failed to execute {order['ticker']}: {e}")
            still_pending.append(order)

    state["deferred_orders"] = still_pending
    return executed


# =========================================================
# DISCOVERY
# =========================================================

def run_discovery(mode, all_tickers, excluded_tickers):
    """Always returns (flagged_dict, scanned_count)."""
    candidates = [t for t in all_tickers if t not in excluded_tickers]
    if MAX_UNIVERSE:
        candidates = candidates[:MAX_UNIVERSE]

    if mode == "news_discovery":
        flagged = discovery.scan_news_discovery(candidates)
        return flagged, len(candidates)

    start_date = pd.Timestamp.today() - pd.Timedelta(days=365 * 5)
    end_date_exclusive = pd.Timestamp.today() + pd.Timedelta(days=1)
    hist_by_ticker = sm.batch_fetch_history(candidates, start_date, end_date_exclusive)

    if mode == "long_horizon":
        return discovery.scan_long_horizon(hist_by_ticker), len(candidates)
    return discovery.scan_core(hist_by_ticker), len(candidates)


# =========================================================
# SIZING
# =========================================================

def size_positions(scored_candidates, account, num_slots):
    """
    scored_candidates: list of dicts with ticker, score, rationale, mode
    Returns the subset to actually buy with a dollar_amount each,
    proportional to score, capped per-position, within a cash budget.
    """
    if num_slots <= 0 or not scored_candidates:
        return []

    ranked = sorted(scored_candidates, key=lambda c: c["score"], reverse=True)[:num_slots]

    equity = account["equity"]
    budget = account["cash"] * (1 - CASH_BUFFER_FRACTION)
    max_per_position = equity * MAX_ALLOCATION_FRACTION

    total_score = sum(c["score"] for c in ranked) or 1.0
    sized = []
    for c in ranked:
        raw_share = budget * (c["score"] / total_score)
        dollar_amount = max(MIN_ORDER_DOLLARS, min(raw_share, max_per_position))
        sized.append({**c, "dollar_amount": dollar_amount})

    # If proportional sizing overshoots the budget (can happen once
    # capping kicks in), scale everything down together.
    total_requested = sum(c["dollar_amount"] for c in sized)
    if total_requested > budget and total_requested > 0:
        scale = budget / total_requested
        for c in sized:
            c["dollar_amount"] = max(MIN_ORDER_DOLLARS, c["dollar_amount"] * scale)

    return [c for c in sized if c["dollar_amount"] >= MIN_ORDER_DOLLARS]


# =========================================================
# MAIN
# =========================================================

def run(state_path="state.json", dry_run=False):
    today = date.today()
    now = datetime.now()
    print(f"=== Agent run: {now.isoformat()} ===")

    trader = AlpacaTrader()
    clock = trader.get_clock()
    account = trader.get_account_summary()
    market_open = clock["is_open"]

    print(f"Market open: {market_open} | Equity: ${account['equity']:,.2f}")

    state = pt.load_state(state_path)

    # --- Execute any deferred orders now that we know market status ---
    executed_deferred = []
    if market_open:
        executed_deferred = execute_deferred_orders(trader, state, today)

    # --- Position monitoring (every run) ---
    held_positions, sells, exit_flagged_waiting = monitor_positions(
        trader, state, today, market_open
    )
    held_tickers = {p["ticker"] for p in held_positions}

    # --- Discovery (rotating mode) ---
    mode = pt.next_mode_and_increment(state, discovery.MODES)
    all_tickers = sm.get_stock_tickers()

    flagged, scanned_count = run_discovery(mode, all_tickers, held_tickers)

    print(f"Mode: {mode} | Scanned {scanned_count} | Flagged {len(flagged)} tickers")

    # --- Sentiment + composite reasoning for flagged tickers ---
    sentiments = ns.get_sentiment_batch(list(flagged.keys()))

    scored_candidates = []
    for ticker, (prob, latest_row) in flagged.items():
        evaluation = reasoning.evaluate(ticker, prob, sentiments.get(ticker), latest_row, mode)
        scored_candidates.append({
            "ticker": ticker,
            "score": evaluation["score"],
            "rationale": evaluation["rationale"],
            "mode": mode,
        })

    # --- Sizing + execution ---
    num_slots = MAX_POSITIONS - len(held_tickers)
    to_buy = size_positions(scored_candidates, account, num_slots)

    new_buys = list(executed_deferred)
    deferred_buys = []

    for candidate in to_buy:
        ticker = candidate["ticker"]
        if market_open:
            try:
                if not dry_run:
                    trader.buy_notional(ticker, candidate["dollar_amount"])
                    pt.record_new_position(
                        state, ticker, today,
                        rationale=candidate["rationale"],
                        mode=candidate["mode"],
                        min_hold_trading_days=MIN_HOLD_TRADING_DAYS,
                    )
                new_buys.append(candidate)
            except Exception as e:
                print(f"[buy] failed for {ticker}: {e}")
        else:
            if not dry_run:
                pt.queue_deferred_order(
                    state, ticker, candidate["dollar_amount"], candidate["rationale"],
                    candidate["mode"], now, clock["next_open"],
                )
            deferred_buys.append({
                **candidate,
                "target_time": clock["next_open"].strftime("%Y-%m-%d %H:%M %Z") if clock["next_open"] else "next open",
            })

    if not dry_run:
        pt.save_state(state, state_path)

    summary = format_run_summary(
        run_date_str=now.strftime("%Y-%m-%d %H:%M"),
        market_open=market_open,
        mode=mode,
        account=account,
        held_positions=held_positions,
        new_buys=new_buys,
        deferred_buys=deferred_buys,
        sells=sells,
        exit_flagged_waiting=exit_flagged_waiting,
        candidates_scanned=scanned_count,
    )
    print(summary)
    send_telegram_message(summary)

    return {
        "held_positions": held_positions,
        "new_buys": new_buys,
        "deferred_buys": deferred_buys,
        "sells": sells,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="state.json")
    parser.add_argument("--dry-run", action="store_true",
                         help="Score and notify but don't submit orders or write state.json")
    args = parser.parse_args()

    run(state_path=args.state, dry_run=args.dry_run)

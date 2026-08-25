"""
Thin wrapper around Alpaca's paper trading API (alpaca-py).

Paper trading on Alpaca is free - sign up at https://alpaca.markets,
generate a *paper* API key/secret pair (Dashboard -> Paper Trading),
and set ALPACA_API_KEY / ALPACA_SECRET_KEY as env vars / repo secrets.

Note on starting balance: Alpaca paper accounts default to $100,000
and the trading API doesn't expose a "set custom starting balance"
call. To start at $10,000 as requested, reset the paper account from
the Alpaca dashboard (Paper Trading -> account menu -> Reset Account,
choose $10,000) before the agent's first run. The agent itself always
reads the *actual* account equity/buying power from Alpaca rather than
hardcoding $10,000, so position sizing stays correct whatever the
dashboard balance is set to.
"""

import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class AlpacaTrader:
    def __init__(self, api_key=None, secret_key=None, paper=True):
        api_key = api_key or os.environ.get("ALPACA_API_KEY")
        secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Create a free "
                "paper trading key at https://alpaca.markets and set them "
                "as env vars / repo secrets."
            )

        self.client = TradingClient(api_key, secret_key, paper=paper)

    # -----------------------------------------------------------------
    # Account / market state
    # -----------------------------------------------------------------

    def get_account_summary(self):
        acct = self.client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
        }

    def get_clock(self):
        """Returns {'is_open': bool, 'next_open': datetime, 'next_close': datetime}."""
        clock = self.client.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": clock.next_open,
            "next_close": clock.next_close,
        }

    # -----------------------------------------------------------------
    # Positions
    # -----------------------------------------------------------------

    def get_positions(self):
        """
        Returns {ticker: {qty, avg_entry_price, current_price,
        market_value, unrealized_plpc}} for everything currently held
        in the Alpaca paper account. This is the source of truth for
        "what do we actually hold" - state.json only stores metadata
        (buy date, rationale, min sell date) alongside it.
        """
        positions = self.client.get_all_positions()
        out = {}
        for p in positions:
            out[p.symbol] = {
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_plpc": float(p.unrealized_plpc),  # e.g. 0.05 = +5%
            }
        return out

    # -----------------------------------------------------------------
    # Orders
    # -----------------------------------------------------------------

    def buy_notional(self, ticker, dollar_amount):
        """
        Market buy order sized in dollars rather than shares (fractional
        shares), so the AI's dollar allocation maps directly to an
        order without doing share-count math. Only fillable while the
        market is open - callers should check get_clock() first.
        """
        order = MarketOrderRequest(
            symbol=ticker,
            notional=round(dollar_amount, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order)

    def sell_all(self, ticker):
        """Closes the entire position in one call."""
        return self.client.close_position(ticker)

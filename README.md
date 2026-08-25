# Stock prediction agent v2 - Alpaca paper trading + rotating discovery

Runs a price model + free news sentiment + technicals, sizes and places
real (paper) trades via Alpaca, tracks a minimum 2-trading-day hold per
position but decides exits on its own after that, and rotates between
three discovery strategies since it now runs every 20-30 minutes.

## What's new vs. the first version

1. **Telegram message now lists current holdings** - ticker, entry
   date, live P/L, and whether it's eligible to sell yet or when it
   will be.
2. **Real paper trading via Alpaca**, free. Orders only execute while
   the market is open; if a buy decision is made while it's closed,
   it's queued as a **deferred order** and shown in its own message
   section, then submitted automatically once the market opens.
3. **Position sizing is dynamic**, not fixed - each buy is sized as a
   share of current account cash, weighted by a composite score
   (model confidence + sentiment + technicals), capped at 20% of
   equity per position and floored at $250 so orders aren't trivially
   small.
4. **Exits are rule-based and independent of the 2-day mark** - a
   stop-loss (-8%) or take-profit (+15%) can fire at any time, but is
   only executed once the 2-trading-day minimum hold has passed.
   Until then it's reported as "flagged, waiting on hold."
5. **Three rotating discovery modes** so each 20-30 minute run isn't
   redoing identical work: `core` (2-day model, same as before),
   `long_horizon` (~10-day model with lighter hyperparameters, aimed
   at longer setups), and `news_discovery` (scans general market news
   for which S&P 500 names are getting unusual headline volume, then
   scores just that shortlist).
6. **Composite reasoning with a written rationale** per candidate -
   free by default (rule-based template), optionally upgraded to an
   actual Claude Haiku-written judgment if you set `ANTHROPIC_API_KEY`
   (small per-call cost, not free - see caveat below).

## Being straight about point 5 (from your last message)

You asked for something that can "make logical decisions on its own,"
not just run the model and buy on a threshold. What's built here is a
multi-factor composite score (model probability + sentiment +
technical trend) with a generated rationale - that's real, and it's
more than the v1 threshold-only logic. But it's still fundamentally
rule-based unless you enable the optional Claude API hook. A system
that's actually reasoning about a thesis, weighing conflicting
evidence, and explaining trade-offs the way a person would needs an
LLM in the loop - that's not free at any real usage volume, though
Haiku is cheap. The default path here works with zero added cost;
flip on `ANTHROPIC_API_KEY` if you want the qualitative upgrade.

## Setup (mostly free)

### 1. Alpaca paper trading account (free)
- Sign up at https://alpaca.markets, go to the Paper Trading dashboard.
- Generate a paper API key/secret pair.
- **Starting balance**: Alpaca paper accounts default to $100,000, and
  the trading API doesn't support setting a custom starting balance
  via code. To start at $10,000 as you wanted, use the dashboard's
  "Reset Account" option and choose $10,000 before the first run. The
  agent always reads your *actual* balance from Alpaca for sizing, so
  it'll work correctly whatever you set it to.

### 2. Telegram bot (free) - same as before
- `@BotFather` -> `/newbot` -> get a token.
- Message the bot once, then hit
  `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat id.

### 3. (Optional, small cost) Anthropic API key
- Only needed if you want the LLM-written rationale instead of the
  free template. Skip this entirely and the agent works fine.

### 4. Repo secrets
Add these under Settings -> Secrets and variables -> Actions:
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- `ANTHROPIC_API_KEY` (optional)

### 5. Schedule
`.github/workflows/agent.yml` runs every 25 minutes, roughly across
market hours in UTC. Adjust the cron if you want tighter/looser
coverage - see the comment in the file about DST drift.

## Local test run

```bash
pip install -r requirements.txt
python agent.py --dry-run    # scores and would-buy/sell but places no orders, doesn't touch state.json
```

## Files

- `stock_model.py` - feature engineering + model training. Now accepts
  a `horizon` parameter so `long_horizon` mode can reuse it with a
  different label window.
- `alpaca_trader.py` - thin wrapper over Alpaca's paper trading API:
  account, market clock, positions, buy (notional/dollar-based),
  sell (closes full position).
- `discovery.py` - the three rotating discovery strategies.
- `reasoning.py` - composite scoring + rationale generation, with the
  optional Claude Haiku hook.
- `news_sentiment.py` - free RSS + VADER sentiment (unchanged from v1).
- `position_tracker.py` - state.json: per-position metadata (entry
  date, min sell date, rationale, mode) and the deferred-order queue.
  Alpaca itself remains the source of truth for what's actually held.
- `notifier.py` - Telegram formatting, now with holdings/deferred/exit
  sections.
- `agent.py` - orchestrates all of the above for one run.

## Known limitations, worth knowing before trusting output

- **Exit rules are simple thresholds** (stop-loss/take-profit on
  unrealized P/L), not a re-evaluation of the original thesis. A
  position could hit neither threshold and just be quietly wrong for
  a long time; there's no "the setup broke" logic beyond price moving
  against you by a set amount.
- **The 2-trading-day hold uses calendar-day math that skips
  weekends**, not true NYSE trading-day/holiday accounting.
- **`long_horizon` mode's ~0.90 probability threshold isn't directly
  comparable to `core`'s 0.95** - they're different models with
  different label distributions, not the same confidence scale.
- **`news_discovery`'s ticker extraction is a crude substring match**
  against known tickers in headline text - it'll miss anything
  referenced only by company name, and can occasionally false-positive
  on short tickers that double as common words/acronyms.
- **Running every 20-30 minutes means ~20-25 runs/day.** Each run
  does a batched full-universe fetch via yfinance in `core`/
  `long_horizon` modes - watch for rate-limiting if you shorten the
  interval further, and consider setting `MAX_UNIVERSE` in `agent.py`
  if runs start taking too long for the Actions time limit.
- **This is paper trading only.** Nothing here places real orders, and
  none of it is financial advice. Historical/backtest behavior,
  including with the earlier leak fix, still doesn't guarantee live
  results - the composite score and exit thresholds are reasonable
  starting points, not tuned/validated defaults.

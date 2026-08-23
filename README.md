# Stock prediction agent (free stack)

Runs your existing price/volume model plus a free news-sentiment veto,
tracks a minimum 2-trading-day hold per position, and sends a Telegram
notification with what it would buy or sell. It does not place real
trades - it only recommends.

## What changed vs. the original script

- **Leak fix**: `score_ticker()` in `agent.py` trains the model on all
  rows *except* the most recent one, then scores that held-out row.
  In the original script, the model was trained on the same row it
  then predicted on, which inflates confidence artificially.
- **Batched price fetch**: one `yf.download()` call for the whole
  candidate universe instead of one request per ticker.
- **News fetched only for model-flagged tickers**: RSS lookups happen
  after the price model has already narrowed the list, not for all
  ~500 tickers every run - keeps runtime and request volume down.

## Setup (all free)

1. **Create a Telegram bot**
   - Message `@BotFather` on Telegram, run `/newbot`, follow the prompts.
   - You'll get a token like `123456789:AAExxxxxxxxxxxxxxxxxxxx`.
   - Send your new bot any message so it knows your chat.
   - Visit `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser
     and copy the `chat.id` value.

2. **Push this folder to a GitHub repo** (public repos get unlimited
   free Actions minutes; private repos get a generous free monthly
   quota).

3. **Add repo secrets** (Settings → Secrets and variables → Actions):
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

4. **Adjust the schedule** in `.github/workflows/agent.yml` if you
   want a different run time - cron doesn't handle DST, so the UTC
   offset will drift by an hour for part of the year.

5. Commit and push. The workflow also has `workflow_dispatch`, so you
   can trigger a manual run from the Actions tab to test before
   waiting for the schedule.

## Local test run

```bash
pip install -r requirements.txt
python agent.py --dry-run    # scores and notifies but doesn't touch state.json
```

## Files

- `stock_model.py` - your original feature engineering + model training,
  unchanged, minus the old sequential live-prediction path (replaced
  by the batched version in `agent.py`).
- `news_sentiment.py` - free RSS headline fetch (Yahoo Finance + Google
  News) and VADER sentiment scoring.
- `position_tracker.py` - JSON-file state, enforces the 2-trading-day
  minimum hold between buy and sell.
- `notifier.py` - Telegram message sending and formatting.
- `agent.py` - orchestrates all of the above for one run.
- `.github/workflows/agent.yml` - free scheduled runner via GitHub Actions.

## Known limitations, worth knowing before trusting output

- **Sentiment is a coarse veto, not a strong signal.** VADER is a
  general-purpose rule-based scorer, not tuned for financial text. It
  can misread sarcasm, negation, or finance-specific phrasing. Treat
  it as "don't buy into an obvious wall of bad headlines," not as
  real signal.
- **`yfinance` data is free but delayed and rate-limited.** Fine for
  a once- or twice-daily signal; not suitable for anything time-sensitive.
- **The 2-day hold uses calendar-day math that skips weekends**, not
  true NYSE trading-day/holiday accounting - it'll be off by a day
  around market holidays.
- **Survivorship bias and the original 2-day feature-label lag**
  (discussed earlier) still apply to the underlying model - this
  agent wraps the model, it doesn't change its statistical properties.
- **This is not financial advice**, and nothing here guarantees the
  model's historical behavior (even after the leak fix) will hold up
  on future, live data. Consider paper-trading the recommendations
  for a while before acting on them with real money.

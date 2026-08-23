"""
Sends notifications via a free Telegram bot.

Setup (one-time, free):
  1. Message @BotFather on Telegram, run /newbot, follow the prompts.
     You'll get a bot token like "123456789:AAExxxxxxxxxxxxxxxxxxxx".
  2. Message your new bot anything (so it can see your chat).
  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates in a browser
     and find "chat":{"id": ...} - that's your chat id.
  4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as env vars / repo
     secrets (see the GitHub Actions workflow).

No paid tier, no rate limit concerns at this message volume.
"""

import os
import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram_message(text, token=None, chat_id=None, parse_mode="Markdown"):
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[notifier] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set - "
              "printing message instead of sending:\n")
        print(text)
        return False

    # Telegram messages cap at 4096 chars - split if needed.
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)] or [text]

    ok = True
    for chunk in chunks:
        resp = requests.post(
            TELEGRAM_API.format(token=token),
            data={
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[notifier] Telegram send failed: {resp.status_code} {resp.text}")
            ok = False

    return ok


def format_run_summary(buys, sells, skipped_count, run_date):
    """
    buys:  list of dicts with keys ticker, model_prob, sentiment, price
    sells: list of dicts with keys ticker, buy_price, sell_price, pct_return
    """
    lines = [f"*Stock agent run - {run_date}*", ""]

    if buys:
        lines.append(f"*New buys ({len(buys)}):*")
        for b in buys:
            sent = "n/a" if b["sentiment"] is None else f"{b['sentiment']:+.2f}"
            lines.append(
                f"- {b['ticker']}: prob {b['model_prob']:.2f}, "
                f"sentiment {sent}, price ${b['price']:.2f}"
            )
    else:
        lines.append("*New buys:* none")

    lines.append("")

    if sells:
        lines.append(f"*Sells ({len(sells)}):*")
        for s in sells:
            lines.append(
                f"- {s['ticker']}: ${s['buy_price']:.2f} -> "
                f"${s['sell_price']:.2f} ({s['pct_return']*100:+.2f}%)"
            )
    else:
        lines.append("*Sells:* none")

    lines.append("")
    lines.append(f"_Skipped {skipped_count} tickers still within their hold period._")
    lines.append("_Not financial advice - model output only._")

    return "\n".join(lines)

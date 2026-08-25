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


def format_run_summary(
    run_date_str,
    market_open,
    mode,
    account,
    held_positions,      # list of dicts: ticker, entry_date, min_sell_date, sellable, unrealized_plpc, market_value, rationale
    new_buys,             # list of dicts: ticker, dollar_amount, score, rationale
    deferred_buys,        # list of dicts: ticker, dollar_amount, rationale, target_time
    sells,                 # list of dicts: ticker, reason, unrealized_plpc, market_value
    exit_flagged_waiting,  # list of dicts: ticker, reason, min_sell_date
    candidates_scanned,
):
    lines = [f"*Stock agent run - {run_date_str}*",
             f"Market: {'OPEN' if market_open else 'CLOSED'} | Discovery mode: `{mode}`",
             f"Account equity: ${account['equity']:,.2f} | Cash: ${account['cash']:,.2f}",
             f"Scanned {candidates_scanned} candidates this run.",
             ""]

    # --- Current holdings ---
    lines.append(f"*Currently held ({len(held_positions)}):*")
    if held_positions:
        for p in held_positions:
            sell_note = "eligible to sell now" if p["sellable"] else f"earliest sell: {p['min_sell_date']}"
            lines.append(
                f"- {p['ticker']}: entered {p['entry_date']}, "
                f"P/L {p['unrealized_plpc']*100:+.2f}%, "
                f"value ${p['market_value']:,.2f}, {sell_note}"
            )
    else:
        lines.append("none")
    lines.append("")

    # --- New buys executed ---
    lines.append(f"*New buys executed ({len(new_buys)}):*")
    if new_buys:
        for b in new_buys:
            score_note = f" (score {b['score']:.2f})" if b.get("score") is not None else ""
            lines.append(f"- {b['ticker']}: ${b['dollar_amount']:,.2f}{score_note}")
            lines.append(f"  _{b['rationale']}_")
    else:
        lines.append("none")
    lines.append("")

    # --- Deferred (market closed) ---
    lines.append(f"*Deferred buys, queued for next open ({len(deferred_buys)}):*")
    if deferred_buys:
        for d in deferred_buys:
            target = d.get("target_time", "next open")
            lines.append(f"- {d['ticker']}: ${d['dollar_amount']:,.2f} planned at {target}")
            lines.append(f"  _{d['rationale']}_")
    else:
        lines.append("none")
    lines.append("")

    # --- Sells executed ---
    lines.append(f"*Sells executed ({len(sells)}):*")
    if sells:
        for s in sells:
            lines.append(
                f"- {s['ticker']}: {s['reason']} "
                f"(P/L {s['unrealized_plpc']*100:+.2f}%, ${s['market_value']:,.2f})"
            )
    else:
        lines.append("none")
    lines.append("")

    # --- Flagged for exit but still within min hold ---
    lines.append(f"*Flagged to exit, waiting on min hold ({len(exit_flagged_waiting)}):*")
    if exit_flagged_waiting:
        for f in exit_flagged_waiting:
            lines.append(f"- {f['ticker']}: {f['reason']} (earliest sell: {f['min_sell_date']})")
    else:
        lines.append("none")
    lines.append("")

    lines.append("_Paper trading only. Not financial advice._")

    return "\n".join(lines)

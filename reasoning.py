"""
Turns the raw signals (model probability, sentiment, technicals) into
a single composite score plus a written rationale.

Be clear-eyed about what this is: a deterministic, explainable
multi-factor weighting, not genuine reasoning. It's the free default.

If you want an actual LLM synthesizing the signals into a written
judgment call (closer to what "decide on its own" implies), set
ANTHROPIC_API_KEY and this module will call Claude Haiku to write
the rationale instead of using the template. That's not free -
Haiku is inexpensive, but each call costs a small amount - so it's
opt-in, and the code falls back to the free template if the key or
the call is missing/fails.
"""

import os
import requests

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# Composite score weights - tune these based on what you find matters
# after watching real results for a while.
WEIGHTS = {
    "model_prob": 0.5,     # price-model confidence, 0-1
    "sentiment": 0.25,     # news sentiment, rescaled from [-1,1] to [0,1]
    "technical": 0.25,     # trend/momentum agreement, 0-1
}


def technical_score(latest_row):
    """
    Rough 0-1 "does price action agree with a bullish read" score from
    already-computed indicator columns (SMA20/50, RSI, MACD). This is
    a simple rule stack, not a separate model.
    """
    score = 0.0
    count = 0

    close = float(latest_row["Close"].iloc[0])

    if "SMA20" in latest_row and "SMA50" in latest_row:
        sma20 = float(latest_row["SMA20"].iloc[0])
        sma50 = float(latest_row["SMA50"].iloc[0])
        score += 1.0 if sma20 > sma50 else 0.0  # short-term uptrend vs long-term
        count += 1

    if "RSI" in latest_row:
        rsi = float(latest_row["RSI"].iloc[0])
        # Reward recovering-from-oversold / healthy momentum, penalize
        # extreme overbought (>80, likely to mean-revert down).
        if rsi < 30:
            score += 0.5
        elif rsi > 80:
            score += 0.0
        else:
            score += 0.75
        count += 1

    if "MACD" in latest_row:
        macd = float(latest_row["MACD"].iloc[0])
        score += 1.0 if macd > 0 else 0.0
        count += 1

    if close and "SMA20" in latest_row:
        sma20 = float(latest_row["SMA20"].iloc[0])
        score += 1.0 if close > sma20 else 0.0
        count += 1

    return score / count if count else 0.5


def composite_score(model_prob, sentiment_compound, tech_score):
    sentiment_component = 0.5 if sentiment_compound is None else (sentiment_compound + 1) / 2

    total = (
        WEIGHTS["model_prob"] * model_prob
        + WEIGHTS["sentiment"] * sentiment_component
        + WEIGHTS["technical"] * tech_score
    )
    return max(0.0, min(1.0, total))


def template_rationale(ticker, model_prob, sentiment_compound, tech_score, mode, headlines):
    parts = [f"Price model: {model_prob:.0%} confidence (mode: {mode})."]

    if sentiment_compound is None:
        parts.append("No recent headlines found, sentiment neutral by default.")
    else:
        tone = "positive" if sentiment_compound > 0.15 else (
            "negative" if sentiment_compound < -0.15 else "mixed/neutral"
        )
        parts.append(f"News sentiment {tone} ({sentiment_compound:+.2f}).")

    trend = "bullish" if tech_score > 0.6 else ("bearish" if tech_score < 0.4 else "mixed")
    parts.append(f"Technicals lean {trend} (score {tech_score:.2f}).")

    if headlines:
        parts.append(f'Sample headline: "{headlines[0][:100]}"')

    return " ".join(parts)


def llm_rationale(ticker, model_prob, sentiment_compound, tech_score, mode, headlines):
    """
    Optional: ask Claude Haiku to synthesize the same signals into a
    short written judgment. Returns None on any failure so the caller
    can fall back to template_rationale.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    prompt = (
        f"You're assisting a paper-trading bot, not giving financial advice. "
        f"Ticker: {ticker}. Quant price model probability of a positive move: "
        f"{model_prob:.2f}. News sentiment (-1 to 1): "
        f"{'n/a' if sentiment_compound is None else round(sentiment_compound, 2)}. "
        f"Technical trend score (0-1, bullish=high): {round(tech_score, 2)}. "
        f"Discovery mode: {mode}. Recent headlines: {headlines[:3]}. "
        f"In 2 short sentences, give a plain-language read on this setup "
        f"and flag the biggest risk to the thesis. No disclaimers, no preamble."
    )

    try:
        resp = requests.post(
            ANTHROPIC_MESSAGES_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json().get("content", [])
        text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
        return text.strip() or None
    except Exception:
        return None


def evaluate(ticker, model_prob, sentiment_info, latest_row, mode):
    """
    Single entry point: returns
      {score, rationale, tech_score, sentiment_compound}
    """
    sentiment_compound = (sentiment_info or {}).get("mean_compound")
    headlines = (sentiment_info or {}).get("headlines", [])
    tech = technical_score(latest_row)
    score = composite_score(model_prob, sentiment_compound, tech)

    rationale = llm_rationale(ticker, model_prob, sentiment_compound, tech, mode, headlines)
    if rationale is None:
        rationale = template_rationale(ticker, model_prob, sentiment_compound, tech, mode, headlines)

    return {
        "score": score,
        "rationale": rationale,
        "tech_score": tech,
        "sentiment_compound": sentiment_compound,
    }

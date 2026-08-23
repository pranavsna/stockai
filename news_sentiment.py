"""
Free news + sentiment scoring.

Sources (no API key required):
  - Yahoo Finance per-ticker RSS feed
  - Google News RSS search, scoped to the ticker

Sentiment:
  - VADER (vaderSentiment package), a free, local, rule-based
    sentiment scorer. It's not as good as a real financial-news
    model, but it's free, fast, and needs no downloads/keys.

This is meant as a coarse FILTER (does recent news skew clearly
negative?), not a primary signal. Headline sentiment is noisy -
don't weight it more than the price-based model.
"""

import time
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_RSS = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"


def _fetch_headlines(ticker, max_items=15, timeout=8):
    """Pull recent headlines for a ticker from both RSS sources."""
    headlines = []

    for url_template in (YAHOO_RSS, GOOGLE_RSS):
        try:
            feed = feedparser.parse(url_template.format(ticker=ticker))
            for entry in feed.entries[:max_items]:
                title = getattr(entry, "title", None)
                if title:
                    headlines.append(title)
        except Exception:
            # A single feed failing shouldn't kill the whole ticker -
            # just fall back to whatever the other source returned.
            continue

    return headlines


def get_sentiment(ticker, max_items=15):
    """
    Returns a dict:
      {
        "ticker": ticker,
        "headline_count": int,
        "mean_compound": float in [-1, 1] or None if no headlines,
        "headlines": [str, ...]   # kept for the notification message
      }

    mean_compound is the average VADER compound score across
    headlines. None means "no recent news found" - the caller
    should decide how to treat that (e.g. neutral / skip).
    """
    headlines = _fetch_headlines(ticker, max_items=max_items)

    if not headlines:
        return {
            "ticker": ticker,
            "headline_count": 0,
            "mean_compound": None,
            "headlines": [],
        }

    scores = [_analyzer.polarity_scores(h)["compound"] for h in headlines]
    mean_compound = sum(scores) / len(scores)

    return {
        "ticker": ticker,
        "headline_count": len(headlines),
        "mean_compound": mean_compound,
        "headlines": headlines[:5],  # keep a short sample for notifications
    }


def get_sentiment_batch(tickers, max_workers=10, max_items=15):
    """
    Fetches sentiment for many tickers concurrently (RSS fetches are
    I/O-bound, so threads help a lot here - this is NOT parallel
    model training, just parallel HTTP requests).

    Returns {ticker: sentiment_dict}.
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(get_sentiment, t, max_items): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                results[ticker] = {
                    "ticker": ticker,
                    "headline_count": 0,
                    "mean_compound": None,
                    "headlines": [],
                    "error": str(e),
                }

    return results

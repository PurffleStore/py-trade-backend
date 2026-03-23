# news_sentiment.py
# pip install gnews nltk rapidfuzz

from __future__ import annotations
from datetime import datetime, timezone
import time
from typing import List, Dict, Any

from gnews import GNews
from rapidfuzz import fuzz
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER is available (safe to call multiple times)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# Keep one analyzer instance
_SIA = SentimentIntensityAnalyzer()


def _sentiment_label(compound: float) -> str:
    if compound > 0.05:
        return "Positive"
    elif compound < -0.05:
        return "Negative"
    return "Neutral"


def _is_similar(title: str, seen_titles: List[str], threshold: int = 60) -> bool:
    for t in seen_titles:
        if fuzz.ratio(title, t) > threshold:
            return True
    return False


def get_latest_news_with_sentiment(
    query: str,
    *,
    period: str = "1d",          
    max_results: int = 20,
    language: str = "en",
    country: str = "US",        
    retries: int = 3,
    backoff_seconds: int = 3
) -> Dict[str, Any]:
   
   
    seen_titles: List[str] = []
    results = []

    for attempt in range(retries):
        try:
            g = GNews(language=language, country=country, period=period, max_results=max_results)
            results = g.get_news(query) or []
            if results:
                break
        except Exception as e:
            print(f"[Attempt {attempt+1}] GNews error: {e}")
        time.sleep(backoff_seconds * (attempt + 1))

    if not results:
        return {"overall_news_score": 0.0, "count": 0, "items": []}

    items: List[Dict[str, Any]] = []
    total_compound = 0.0

    for art in results:
        title = (art.get("title") or "").strip()
        if not title:
            continue
        if _is_similar(title, seen_titles, threshold=60):
            continue
        seen_titles.append(title)

        url = (art.get("url")
               or art.get("link")
               or art.get("source", {}).get("url")
               or "")

        published_raw = (art.get("published date")
                         or art.get("publishedDate")
                         or art.get("datetime")
                         or "")
        if isinstance(published_raw, datetime):
            if published_raw.tzinfo is None:
                published_raw = published_raw.replace(tzinfo=timezone.utc)
            published = published_raw.strftime("%Y-%m-%d %H:%M")
        else:
            published = str(published_raw)

        compound = _SIA.polarity_scores(title)["compound"]
        items.append({
            "title": title,
            "url": url,
            "published": published,
            "sentiment": _sentiment_label(compound),
            "compound": round(compound, 3),
        })
        total_compound += compound

    n = len(items)
    overall = round(((total_compound / n) + 1) * 2.5, 2) if n else 0.0

    return {"overall_news_score": overall, "count": n, "items": items}

"""
MSA — Market Sentiment Analyzer
FastAPI backend for stock analysis deployed on Hugging Face Spaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import yfinance as yf
from bs4 import BeautifulSoup
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
CACHE_TTL_SECONDS: int = 300

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("msa")

app = FastAPI(
    title="MSA — Market Sentiment Analyzer",
    description=(
        "Stock analysis API: historical data, sentiment, news, and GPT-4 insights. "
        "Results are cached for 5 minutes per ticker."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_analysis_cache: TTLCache[str, dict[str, Any]] = TTLCache(
    maxsize=256, ttl=CACHE_TTL_SECONDS
)


class MovingAverages(BaseModel):
    sma_50: float | None = Field(None, description="50-day Simple Moving Average")
    sma_200: float | None = Field(None, description="200-day Simple Moving Average")
    signal: str | None = Field(
        None,
        description="'Golden Cross' if SMA-50 > SMA-200, 'Death Cross' otherwise",
    )


class FearGreed(BaseModel):
    value: int = Field(50, description="Fear & Greed index (0-100)")
    label: str = Field("Neutral", description="Human-readable label, e.g. 'Greed'")


class NewsHeadline(BaseModel):
    title: str
    publisher: str | None = None
    link: str | None = None
    published: str | None = None


class TechnicalIndicators(BaseModel):
    rsi_14: float | None = Field(None, description="14-day RSI")
    rsi_signal: str | None = Field(None, description="Oversold / Neutral / Overbought")
    macd: float | None = Field(None, description="MACD line value")
    macd_signal: float | None = Field(None, description="MACD signal line value")
    macd_histogram: float | None = Field(None, description="MACD histogram")
    macd_trend: str | None = Field(None, description="Bullish / Bearish")
    overall_signal: str | None = Field(None, description="Buy / Sell / Neutral")
    score: int = Field(50, description="0-100 gauge score: 0=Strong Sell, 100=Strong Buy")


class AnalystRating(BaseModel):
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    total: int = 0
    recommendation: str | None = Field(None, description="Overall recommendation label")
    score: int = Field(50, description="0-100 gauge score: 0=Strong Sell, 100=Strong Buy")


class PriceTarget(BaseModel):
    current: float | None = None
    target_mean: float | None = None
    target_high: float | None = None
    target_low: float | None = None
    upside_pct: float | None = Field(None, description="% upside to mean target")


class GPTInsight(BaseModel):
    actionable_insight: str = Field(..., description="GPT-4 actionable recommendation")
    confidence_score: int = Field(
        ..., ge=0, le=100, description="Confidence score 0-100"
    )
    reasoning: str = Field(..., description="Brief reasoning behind the recommendation")


class AnalysisResponse(BaseModel):
    ticker: str
    timestamp: str
    cached: bool = Field(False, description="True if this result came from cache")
    moving_averages: MovingAverages
    technicals: TechnicalIndicators | None = None
    analyst_ratings: AnalystRating | None = None
    price_target: PriceTarget | None = None
    fear_greed: FearGreed
    news: list[NewsHeadline]
    gpt_analysis: GPTInsight | None = None


def _yf_fetch_history(ticker: str, period: str = "1y") -> list[dict[str, Any]]:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    rows: list[dict[str, Any]] = []
    for date, row in df.iterrows():
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            }
        )

    rows.reverse()
    return rows


async def fetch_daily_ohlcv(ticker: str) -> list[dict[str, Any]]:
    try:
        return await asyncio.to_thread(_yf_fetch_history, ticker.upper())
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch historical data for {ticker}: {exc}",
        ) from exc


def calculate_smas(daily_rows: list[dict[str, Any]]) -> MovingAverages:
    closes = [r["close"] for r in daily_rows]

    sma_50: float | None = None
    sma_200: float | None = None

    if len(closes) >= 50:
        sma_50 = round(sum(closes[:50]) / 50, 4)
    if len(closes) >= 200:
        sma_200 = round(sum(closes[:200]) / 200, 4)

    signal: str | None = None
    if sma_50 is not None and sma_200 is not None:
        signal = "Golden Cross" if sma_50 > sma_200 else "Death Cross"

    return MovingAverages(sma_50=sma_50, sma_200=sma_200, signal=signal)


def calculate_technicals(daily_rows: list[dict[str, Any]]) -> TechnicalIndicators:
    closes = [r["close"] for r in daily_rows]

    # RSI-14
    rsi_14: float | None = None
    rsi_signal: str | None = None
    if len(closes) >= 15:
        gains, losses = [], []
        for i in range(1, 15):
            delta = closes[i - 1] - closes[i]  # rows are newest-first
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss == 0:
            rsi_14 = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14 = round(100 - (100 / (1 + rs)), 2)
        if rsi_14 <= 30:
            rsi_signal = "Oversold"
        elif rsi_14 >= 70:
            rsi_signal = "Overbought"
        else:
            rsi_signal = "Neutral"

    # MACD (12, 26, 9) — we need at least ~35 days of data
    macd_val: float | None = None
    macd_sig: float | None = None
    macd_hist: float | None = None
    macd_trend: str | None = None

    if len(closes) >= 35:
        ordered = list(reversed(closes))  # oldest-first for EMA calc

        def ema(data: list[float], span: int) -> list[float]:
            k = 2 / (span + 1)
            result = [data[0]]
            for price in data[1:]:
                result.append(price * k + result[-1] * (1 - k))
            return result

        ema12 = ema(ordered, 12)
        ema26 = ema(ordered, 26)
        macd_line = [a - b for a, b in zip(ema12[25:], ema26[25:])]

        if len(macd_line) >= 9:
            signal_line = ema(macd_line, 9)
            macd_val = round(macd_line[-1], 4)
            macd_sig = round(signal_line[-1], 4)
            macd_hist = round(macd_val - macd_sig, 4)
            macd_trend = "Bullish" if macd_hist > 0 else "Bearish"

    # gauge score: map RSI + MACD into 0-100 (0=Strong Sell, 100=Strong Buy)
    # RSI contributes: <20 → +2, 20-30 → +1, 30-70 → 0, 70-80 → -1, >80 → -2
    rsi_pts = 0
    if rsi_14 is not None:
        if rsi_14 < 20: rsi_pts = 2
        elif rsi_14 < 30: rsi_pts = 1
        elif rsi_14 > 80: rsi_pts = -2
        elif rsi_14 > 70: rsi_pts = -1

    macd_pts = 0
    if macd_hist is not None:
        if macd_hist > 0: macd_pts = 1
        else: macd_pts = -1

    raw = rsi_pts + macd_pts  # range [-3, +3]
    gauge = int(((raw + 3) / 6) * 100)
    gauge = max(0, min(100, gauge))

    if gauge >= 70:
        overall = "Buy" if gauge < 85 else "Strong Buy"
    elif gauge <= 30:
        overall = "Sell" if gauge > 15 else "Strong Sell"
    else:
        overall = "Neutral"

    return TechnicalIndicators(
        rsi_14=rsi_14,
        rsi_signal=rsi_signal,
        macd=macd_val,
        macd_signal=macd_sig,
        macd_histogram=macd_hist,
        macd_trend=macd_trend,
        overall_signal=overall,
        score=gauge,
    )


def _yf_fetch_analyst_ratings(ticker: str) -> AnalystRating:
    try:
        stock = yf.Ticker(ticker.upper())
        rec = stock.recommendations
        if rec is None or rec.empty:
            return AnalystRating()

        latest = rec.iloc[-1]
        sb = int(latest.get("strongBuy", 0))
        b = int(latest.get("buy", 0))
        h = int(latest.get("hold", 0))
        s = int(latest.get("sell", 0))
        ss = int(latest.get("strongSell", 0))
        total = sb + b + h + s + ss

        if total == 0:
            label = "No Data"
        else:
            buy_pct = (sb + b) / total
            sell_pct = (s + ss) / total
            if buy_pct >= 0.6:
                label = "Strong Buy" if sb > b else "Buy"
            elif sell_pct >= 0.6:
                label = "Strong Sell" if ss > s else "Sell"
            else:
                label = "Hold"

        # weighted score: strong_buy=5, buy=4, hold=3, sell=2, strong_sell=1
        if total > 0:
            weighted = (sb * 5 + b * 4 + h * 3 + s * 2 + ss * 1) / total
            a_score = int(((weighted - 1) / 4) * 100)
            a_score = max(0, min(100, a_score))
        else:
            a_score = 50

        return AnalystRating(
            strong_buy=sb, buy=b, hold=h, sell=s, strong_sell=ss,
            total=total, recommendation=label, score=a_score,
        )
    except Exception as exc:
        logger.warning("Analyst ratings fetch failed for %s: %s", ticker, exc)
        return AnalystRating()


def _yf_fetch_price_target(ticker: str) -> PriceTarget:
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info or {}
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low = info.get("targetLowPrice")

        upside = None
        if current and target_mean:
            upside = round(((target_mean - current) / current) * 100, 2)

        return PriceTarget(
            current=round(float(current), 2) if current else None,
            target_mean=round(float(target_mean), 2) if target_mean else None,
            target_high=round(float(target_high), 2) if target_high else None,
            target_low=round(float(target_low), 2) if target_low else None,
            upside_pct=upside,
        )
    except Exception as exc:
        logger.warning("Price target fetch failed for %s: %s", ticker, exc)
        return PriceTarget()


async def fetch_analyst_ratings(ticker: str) -> AnalystRating:
    return await asyncio.to_thread(_yf_fetch_analyst_ratings, ticker.upper())


async def fetch_price_target(ticker: str) -> PriceTarget:
    return await asyncio.to_thread(_yf_fetch_price_target, ticker.upper())


FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_NEUTRAL_FALLBACK = FearGreed(value=50, label="Neutral")


def _fg_label(value: int) -> str:
    if value <= 25:
        return "Extreme Fear"
    if value <= 45:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 75:
        return "Greed"
    return "Extreme Greed"


async def fetch_fear_greed() -> FearGreed:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(FEAR_GREED_URL, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        fg = data.get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating")

        if score is not None:
            return FearGreed(value=int(round(score)), label=rating or _fg_label(int(round(score))))
    except Exception as exc:
        logger.warning("Fear & Greed primary fetch failed: %s", exc)

    try:
        fallback_url = "https://money.cnn.com/data/fear-and-greed/"
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(fallback_url, headers=headers)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        needle = soup.find("div", id="needleChart")
        if needle:
            score_text = needle.find("li")
            if score_text:
                val = int("".join(c for c in score_text.text if c.isdigit())[:3])
                return FearGreed(value=val, label=_fg_label(val))
    except Exception as exc2:
        logger.warning("Fear & Greed fallback failed: %s", exc2)

    logger.warning("Returning Neutral (50) default for Fear & Greed")
    return _NEUTRAL_FALLBACK


def _yf_fetch_news(ticker: str, max_items: int = 10) -> list[NewsHeadline]:
    try:
        stock = yf.Ticker(ticker.upper())
        raw_news = stock.news or []
    except Exception as exc:
        logger.warning("yfinance news error for %s: %s", ticker, exc)
        return []

    headlines: list[NewsHeadline] = []
    for item in raw_news[:max_items]:
        content = item.get("content", item)

        title = content.get("title", "")
        publisher = None
        provider = content.get("provider")
        if isinstance(provider, dict):
            publisher = provider.get("displayName")

        link = None
        canonical = content.get("canonicalUrl") or content.get("clickThroughUrl")
        if isinstance(canonical, dict):
            link = canonical.get("url")

        published_iso = content.get("pubDate")

        if title:
            headlines.append(
                NewsHeadline(
                    title=title,
                    publisher=publisher,
                    link=link,
                    published=published_iso,
                )
            )
    return headlines


async def fetch_news(ticker: str, max_items: int = 10) -> list[NewsHeadline]:
    return await asyncio.to_thread(_yf_fetch_news, ticker.upper(), max_items)


async def fetch_stock_data(
    ticker: str,
) -> tuple[list[dict[str, Any]], list[NewsHeadline]]:
    history, news = await asyncio.gather(
        fetch_daily_ohlcv(ticker),
        fetch_news(ticker),
    )
    return history, news


async def gpt_analysis(
    ticker: str,
    smas: MovingAverages,
    technicals: TechnicalIndicators,
    analyst: AnalystRating,
    fear_greed: FearGreed,
    headlines: list[NewsHeadline],
) -> GPTInsight:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured.")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    news_text = "\n".join(
        f"- {h.title} ({h.publisher})" for h in headlines
    ) or "No recent headlines available."

    payload = {
        "ticker": ticker.upper(),
        "moving_averages": {
            "sma_50": smas.sma_50,
            "sma_200": smas.sma_200,
            "signal": smas.signal,
        },
        "technical_indicators": {
            "rsi_14": technicals.rsi_14,
            "rsi_signal": technicals.rsi_signal,
            "macd": technicals.macd,
            "macd_signal": technicals.macd_signal,
            "macd_histogram": technicals.macd_histogram,
            "macd_trend": technicals.macd_trend,
            "overall_technical_signal": technicals.overall_signal,
        },
        "analyst_ratings": {
            "strong_buy": analyst.strong_buy,
            "buy": analyst.buy,
            "hold": analyst.hold,
            "sell": analyst.sell,
            "strong_sell": analyst.strong_sell,
            "recommendation": analyst.recommendation,
        },
        "fear_and_greed": {
            "value": fear_greed.value,
            "label": fear_greed.label,
        },
        "recent_news_headlines": news_text,
    }

    system_prompt = (
        "You are a JSON-only response engine. Do not include markdown formatting "
        "or prose outside the JSON object.\n\n"
        "Role: Senior Quantitative Analyst at a top-tier hedge fund.\n\n"
        "You will receive a JSON object containing:\n"
        "1. 50-day and 200-day Simple Moving Averages with crossover signal.\n"
        "2. Technical indicators: RSI-14 with overbought/oversold signal, "
        "MACD with trend direction.\n"
        "3. Wall Street analyst ratings breakdown (strongBuy/buy/hold/sell/strongSell).\n"
        "4. The CNN Fear & Greed Index score and label.\n"
        "5. Recent news headlines.\n\n"
        "Your task:\n"
        "- Synthesize ALL signals: SMA crossover, RSI, MACD, analyst consensus, "
        "Fear & Greed sentiment, and news headlines.\n"
        "- Provide a final Actionable Insight: one of Buy / Hold / Sell / Watch "
        "with a concise explanation.\n"
        "- Provide a Confidence Score from 0 to 100.\n\n"
        "Respond with ONLY this JSON schema:\n"
        "{\n"
        '  "actionable_insight": "<Buy|Hold|Sell|Watch> — short explanation",\n'
        '  "confidence_score": <int 0-100>,\n'
        '  "reasoning": "<2-3 sentence justification>"\n'
        "}"
    )

    user_message = (
        "Analyze the following market data and produce your recommendation:\n\n"
        + json.dumps(payload, indent=2)
    )

    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_tokens=600,
    )

    raw = (completion.choices[0].message.content or "{}").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("GPT-4 returned unparseable JSON: %s", raw)
        parsed = {
            "actionable_insight": raw[:500],
            "confidence_score": 0,
            "reasoning": "Failed to parse structured response from GPT-4.",
        }

    return GPTInsight(
        actionable_insight=parsed.get("actionable_insight", "N/A"),
        confidence_score=max(0, min(100, int(parsed.get("confidence_score", 0)))),
        reasoning=parsed.get("reasoning", ""),
    )


@app.get("/")
async def root():
    return {
        "service": "MSA — Market Sentiment Analyzer",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(_analysis_cache)}


@app.get("/analyze", response_model=AnalysisResponse)
async def analyze(
    ticker: str = Query(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol, e.g. AAPL",
    ),
):
    ticker = ticker.upper().strip()

    if ticker in _analysis_cache:
        logger.info("Cache HIT for %s", ticker)
        cached_data = _analysis_cache[ticker].copy()
        cached_data["cached"] = True
        return AnalysisResponse(**cached_data)

    logger.info("Cache MISS — starting analysis for %s", ticker)
    t0 = time.perf_counter()

    (daily_rows, news), fg, analyst, pt = await asyncio.gather(
        fetch_stock_data(ticker),
        fetch_fear_greed(),
        fetch_analyst_ratings(ticker),
        fetch_price_target(ticker),
    )

    smas = calculate_smas(daily_rows)
    technicals = calculate_technicals(daily_rows)

    gpt_result: GPTInsight | None = None
    try:
        gpt_result = await gpt_analysis(ticker, smas, technicals, analyst, fg, news)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("GPT-4 analysis failed: %s", exc)

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("Analysis for %s completed in %.2fs", ticker, elapsed)

    result = AnalysisResponse(
        ticker=ticker,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=False,
        moving_averages=smas,
        technicals=technicals,
        analyst_ratings=analyst,
        price_target=pt,
        fear_greed=fg,
        news=news,
        gpt_analysis=gpt_result,
    )

    _analysis_cache[ticker] = result.model_dump()

    return result


@app.get("/sma", response_model=MovingAverages)
async def sma_only(
    ticker: str = Query(..., min_length=1, max_length=10),
):
    daily = await fetch_daily_ohlcv(ticker.upper().strip())
    return calculate_smas(daily)


@app.get("/fear-greed", response_model=FearGreed)
async def fear_greed_only():
    return await fetch_fear_greed()


@app.get("/news", response_model=list[NewsHeadline])
async def news_only(
    ticker: str = Query(..., min_length=1, max_length=10),
):
    return await fetch_news(ticker.upper().strip())


@app.delete("/cache/{ticker}")
async def clear_cache(ticker: str):
    key = ticker.upper().strip()
    removed = _analysis_cache.pop(key, None) is not None
    return {"ticker": key, "removed": removed, "cache_size": len(_analysis_cache)}


@app.delete("/cache")
async def clear_all_cache():
    count = len(_analysis_cache)
    _analysis_cache.clear()
    return {"cleared": count}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

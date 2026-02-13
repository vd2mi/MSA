# MSA — Market Sentiment Analyzer

AI-powered stock analysis platform that combines technical indicators, market sentiment, and real-time news into a single actionable signal — built for speed, built for traders.

## What It Does

- **Technical Analysis** — Calculates 50-day & 200-day Simple Moving Averages, detects Golden Cross / Death Cross signals
- **Market Sentiment** — Pulls the CNN Fear & Greed Index in real time (falls back to Neutral if unavailable)
- **News Intelligence** — Aggregates latest headlines from Yahoo Finance for any ticker
- **GPT-4o Synthesis** — Fuses all data through a Senior Quant Analyst persona to produce an actionable insight + confidence score (0–100)
- **5-Minute TTL Cache** — Repeat requests are served instantly, saving API costs and preventing rate limits
- **Live Ticker Feed** — WebSocket-powered real-time price belt on the landing page via external data stream

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn, Pydantic |
| Data | yfinance (no rate limits), httpx, BeautifulSoup4 |
| AI | OpenAI GPT-4o (strict JSON mode) |
| Caching | cachetools TTLCache (in-memory, 5 min) |
| Async | asyncio.gather + asyncio.to_thread for non-blocking I/O |
| Frontend | Tailwind CSS, Chart.js, Three.js, GSAP |
| Live Data | WebSocket (wss://fisi-production.up.railway.app) |
| Deployment | Docker on Hugging Face Spaces |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check + cache size |
| `GET` | `/analyze?ticker=AAPL` | **Full pipeline** — SMA + Fear & Greed + News + GPT-4o (cached 5 min) |
| `GET` | `/sma?ticker=AAPL` | SMA data only |
| `GET` | `/fear-greed` | CNN Fear & Greed Index |
| `GET` | `/news?ticker=AAPL` | Yahoo Finance news headlines |
| `DELETE` | `/cache/{ticker}` | Evict one ticker from cache |
| `DELETE` | `/cache` | Purge entire cache |

## Project Structure

```
MSA/
├── app.py              # FastAPI application (all backend logic)
├── index.html          # Landing page (Three.js animation, live ticker belt, stats)
├── dashboard.html      # Analysis terminal (dynamic search, charts, GPT results)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Hugging Face Spaces deployment
├── .env                # Local env vars (not committed)
└── README.md
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key with GPT-4 access |

## Deploying to Hugging Face Spaces

1. Create a new Space → select **Docker** as the SDK.
2. Push this repo to the Space.
3. Add `OPENAI_API_KEY` as a Secret in the Space settings.
4. The Space builds and exposes the API on port **7860**.

Live API: `https://vd2mi-msa.hf.space`

## Local Development

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

Run the server:
```bash
python app.py
```

Server starts on `http://localhost:7860`. Interactive docs at `/docs`.

## Frontend

- **index.html** — Landing page with Three.js market wave animation, WebSocket-powered live ticker belt, and API health stats
- **dashboard.html** — Analysis terminal where users enter any ticker and get full SMA + sentiment + GPT-4o breakdown with charts

Both pages connect to the Hugging Face-hosted API at `https://vd2mi-msa.hf.space`.

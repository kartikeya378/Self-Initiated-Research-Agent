
# Self-Initiated Research Agent (MVP)

A minimal, **end-to-end** prototype that:
1) Fetches recent papers from arXiv for a query
2) Stores them in SQLite
3) Computes embeddings
4) Clusters into topics
5) Detects trending topics
6) Summarizes a trending topic
7) Generates a Markdown report

> This is intentionally lightweight and local-first. No external LLM keys needed.

---

## Quick Start

```bash
# 1) (Optional) Create a virtual env
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure your query (optional)
#    Edit config.yaml to change search query, time windows, etc.

# 4) Run the full pipeline
python main.py run
```

### Useful individual steps

```bash
python main.py fetch    # fetch latest arXiv papers
python main.py embed    # compute sentence embeddings
python main.py topics   # cluster into topics via MiniBatchKMeans
python main.py trends   # detect trending topics
python main.py report   # write a markdown report to ./reports
```

### Where data goes?
- SQLite DB: `./data/agent.db`
- Reports: `./reports/report_*.md`

### Config knobs (config.yaml)
- `query`: arXiv search query
- `days_back`: days of papers to collect per run
- `max_results`: max #results from arXiv per fetch
- `min_topics`/`max_topics`: bounds for k-means (sqrt(n) heuristic inside)
- `min_count`/`ratio_thresh`: trend detection thresholds
- `lookback_recent_days`/`lookback_baseline_days`: windows for spike detection

---

## Notes
- If you run this the very first time, NLTK may download a small tokenizer model.
- For embeddings we use `sentence-transformers` (MiniLM), a small, fast model that runs on CPU.
- You can extend summaries with any LLM/RAG stack later.

# Deploying the FORESIGHT Scoring Service (D6)

Two free options, both work well for this. Render is simpler for a plain FastAPI
service; Hugging Face Spaces is simpler if you're already using it for the Streamlit
dashboard variant.

---

## Option A — Render (recommended for a pure API)

1. Push your `foresight/` repo to GitHub (must include `service/main.py`,
   `service/requirements.txt`, and `data/processed/risk_scores.csv`).
2. Go to [render.com](https://render.com) → sign up free → **New > Web Service**.
3. Connect your GitHub repo.
4. Configure:
   - **Root directory:** leave blank (repo root)
   - **Build command:** `pip install -r service/requirements.txt`
   - **Start command:** `uvicorn service.main:app --host 0.0.0.0 --port $PORT`
   - **Instance type:** Free
5. Click **Create Web Service**. Render builds and deploys automatically.
6. Once live, your public URL looks like `https://foresight-xxxx.onrender.com`.
   Test it: visit `https://foresight-xxxx.onrender.com/docs` — you should see the
   same interactive Swagger UI you saw locally.

**Free-tier note:** Render's free instances sleep after 15 minutes of inactivity and
take ~30–60 seconds to wake on the next request. Mention this in your README so
graders aren't confused by a slow first request.

---

## Option B — Hugging Face Spaces (Docker SDK)

1. Create a file `Dockerfile` in your repo root:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r service/requirements.txt
EXPOSE 7860
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

2. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
3. Choose **Docker** as the Space SDK, make it Public.
4. Push your repo to the Space's git remote (HF gives you the exact git commands
   after creating the Space).
5. The Space builds automatically. Your public URL will be
   `https://huggingface.co/spaces/<your-username>/<space-name>`, with the API
   itself reachable at that same domain on port 7860 internally (HF proxies it).

---

## Verifying the deployment (do this before submitting)

Once deployed, check all of these from a browser or `curl` — not just `/docs`:

```bash
curl https://<your-deployed-url>/health
curl https://<your-deployed-url>/forecast/SKU0057
curl https://<your-deployed-url>/forecast/SKU9999      # should 404 cleanly, not crash
curl -X POST https://<your-deployed-url>/forecast/batch \
  -H "Content-Type: application/json" \
  -d '{"sku_ids": ["SKU0001", "SKU0057", "BADID"]}'
```

If `/health` returns `{"status": "degraded", ...}`, the deployed instance can't find
`risk_scores.csv` — check that the CSV was actually committed to your repo and the
path in `service/main.py` (`data/processed/risk_scores.csv`, relative to repo root)
matches your deployed folder structure.

---

## What to put in your README (Section 13 requirement)

Your top-level `README.md` should include:
- The live URL from whichever option you chose
- The 3 example `curl` commands above, so a grader can test it in 10 seconds
- A one-line note: "Data refreshes by re-running `src/risk_scoring.py` and redeploying —
  this is a static snapshot service, not a live-training system, per the brief's
  out-of-scope Section 4.3 (no real-time pipelines required)."

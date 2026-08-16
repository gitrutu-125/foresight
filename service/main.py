"""
FORESIGHT scoring service
----------------------------
A small API that returns the forecast + risk score for any SKU (or a
batch of SKUs) on demand. This is D6 from the brief: hosted, documented,
handles bad input gracefully.

Run locally:
    uvicorn service.main:app --reload --port 8000

Then visit:
    http://127.0.0.1:8000/docs   (interactive Swagger docs, auto-generated)

Deployment: see reports/deployment_guide.md for Render / Hugging Face
Spaces / Streamlit Cloud instructions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(
    title="FORESIGHT Scoring Service",
    description=(
        "Returns demand forecast and stockout/overstock risk for NorthBay "
        "Living SKUs. Backed by the risk_scores.csv produced by "
        "src/risk_scoring.py — re-run that pipeline and restart this "
        "service to refresh the scores."
    ),
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "risk_scores.csv"

_risk_df: Optional[pd.DataFrame] = None


def get_data() -> pd.DataFrame:
    """Lazy-load the risk scores so the app still starts even if the
    file is briefly missing (e.g. mid-redeploy), and errors are raised
    per-request instead of crashing the whole service."""
    global _risk_df
    if _risk_df is None:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Risk score data not found at {DATA_PATH}. "
                       f"Run src/risk_scoring.py first.",
            )
        _risk_df = pd.read_csv(DATA_PATH)
    return _risk_df


# ---------------------------------------------------------------------
# Response schema (documents the output shape automatically via /docs)
# ---------------------------------------------------------------------
class SkuScore(BaseModel):
    sku_id: str
    category: str
    avg_weekly_demand: float = Field(..., description="Forecasted average weekly demand over the horizon")
    demand_over_horizon: float = Field(..., description="Total forecasted demand over the full horizon")
    available_stock: float = Field(..., description="on_hand_units + on_order_units")
    stockout_risk: float = Field(..., ge=0, le=1)
    overstock_risk: float = Field(..., ge=0, le=1)
    risk_quadrant: str
    recommended_action: str
    rupee_value_at_stake: float


class BatchRequest(BaseModel):
    sku_ids: List[str] = Field(..., min_items=1, max_items=200, example=["SKU0001", "SKU0057"])


class BatchResponse(BaseModel):
    found: List[SkuScore]
    not_found: List[str]


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root():
    return {
        "service": "FORESIGHT Scoring Service",
        "endpoints": ["/health", "/forecast/{sku_id}", "/forecast/batch (POST)", "/docs"],
    }


@app.get("/health", tags=["meta"])
def health():
    try:
        df = get_data()
        return {"status": "ok", "skus_loaded": len(df)}
    except HTTPException as e:
        return {"status": "degraded", "detail": e.detail}


@app.get("/forecast/{sku_id}", response_model=SkuScore, tags=["forecast"])
def get_forecast(sku_id: str):
    """Return forecast + risk for a single SKU. Returns a clean 404
    (not a crash) if the SKU doesn't exist."""
    df = get_data()
    sku_id = sku_id.strip().upper()
    row = df[df["sku_id"].str.upper() == sku_id]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{sku_id}' not found. It may be a new SKU with no sales "
                   f"history yet, or the ID may be mistyped.",
        )

    r = row.iloc[0]
    return SkuScore(
        sku_id=r["sku_id"], category=r["category"],
        avg_weekly_demand=round(float(r["avg_weekly_demand"]), 2),
        demand_over_horizon=round(float(r["demand_over_horizon"]), 2),
        available_stock=round(float(r["available_stock"]), 2),
        stockout_risk=round(float(r["stockout_risk"]), 3),
        overstock_risk=round(float(r["overstock_risk"]), 3),
        risk_quadrant=r["risk_quadrant"],
        recommended_action=r["recommended_action"],
        rupee_value_at_stake=round(float(r["rupee_value_at_stake"]), 2),
    )


@app.post("/forecast/batch", response_model=BatchResponse, tags=["forecast"])
def get_forecast_batch(request: BatchRequest):
    """Return forecast + risk for a list of SKUs in one call. SKUs that
    don't exist are reported in `not_found` rather than failing the
    whole request."""
    df = get_data()
    found, not_found = [], []

    for sku_id in request.sku_ids:
        clean_id = sku_id.strip().upper()
        row = df[df["sku_id"].str.upper() == clean_id]
        if row.empty:
            not_found.append(sku_id)
            continue
        r = row.iloc[0]
        found.append(SkuScore(
            sku_id=r["sku_id"], category=r["category"],
            avg_weekly_demand=round(float(r["avg_weekly_demand"]), 2),
            demand_over_horizon=round(float(r["demand_over_horizon"]), 2),
            available_stock=round(float(r["available_stock"]), 2),
            stockout_risk=round(float(r["stockout_risk"]), 3),
            overstock_risk=round(float(r["overstock_risk"]), 3),
            risk_quadrant=r["risk_quadrant"],
            recommended_action=r["recommended_action"],
            rupee_value_at_stake=round(float(r["rupee_value_at_stake"]), 2),
        ))

    return BatchResponse(found=found, not_found=not_found)


@app.get("/skus", tags=["meta"])
def list_skus(category: Optional[str] = None, risk_quadrant: Optional[str] = None):
    """List available SKU IDs, optionally filtered — useful for exploring
    what's available before calling /forecast/{sku_id}."""
    df = get_data()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if risk_quadrant:
        df = df[df["risk_quadrant"].str.lower() == risk_quadrant.lower()]
    return {"count": len(df), "sku_ids": df["sku_id"].tolist()}

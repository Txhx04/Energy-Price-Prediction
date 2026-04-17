"""
backend/api.py — ML Energy Prediction & Optimization API
=========================================================
REST API server responsible for serving ensemble predictions via HTTP endpoints.
Single FastAPI backend serving:
  - 24h predictions (all 8 models + ensemble)
  - Live forecasts via ENTSO-E API
  - Activity scheduling with PuLP optimizer
  - Validation/test results & feature importances
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys
import csv
from pathlib import Path
from datetime import datetime, timedelta

# Load .env from project root
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Change to project root directory so model paths resolve
os.chdir(PROJECT_ROOT)

from backend.models import ModelManager
from backend.optimizer import (
    generate_optimal_plan, Activity, DEFAULT_ACTIVITIES
)

app = FastAPI(
    title="ML Energy Prediction & Optimization API",
    description="API for 24-hour electricity consumption/price prediction and smart activity scheduling",
    version="2.0.0"
)

# ─────────────────────────────────────────
# CORS Configuration
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Global Model Manager (loaded on startup)
# ─────────────────────────────────────────
model_manager = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize API and load ML models into CPU/GPU memory paths.
    """
    global model_manager
    print("Starting up API server...")
    try:
        model_manager = ModelManager()
        model_manager.load_all_models()
        model_manager.load_data()
        print("[OK] Models and data loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ─────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────
class PredictionRequest(BaseModel):
    date: str

class ActivityInput(BaseModel):
    name: str
    power_kw: float
    duration_hours: int
    priority: int
    earliest_start: int
    latest_finish: int
    icon: str = "⚡"

class OptimizationRequest(BaseModel):
    activities: list[ActivityInput]
    date: str | None = None
    model: str | None = "Ensemble"


# ═════════════════════════════════════════════════════════════
# CORE ENDPOINTS
# ═════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": model_manager is not None and model_manager.models is not None,
        "entsoe_api_key_set": bool(os.environ.get("ENTSOE_API_KEY", "")),
    }


@app.get("/available-dates")
async def get_available_dates():
    """Get available dates from test set."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        return model_manager.get_available_dates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """
    Prediction endpoint handler for historical test-set inferences.
    
    Inputs:
      - request: PredictionRequest JSON body with `date` string (e.g., "2026-01-05").
      
    Outputs:
      - JSON response containing 24-hour step-ahead forecasts from 8 models + ensemble metrics.
      
    ML Principle: Batch processing for 24-hour-ahead forecasting inference.
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        try:
            datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        result = model_manager.run_prediction(request.date)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════
# ENERGY PLANNER ENDPOINTS
# ═════════════════════════════════════════════════════════════

@app.get("/api/live-forecast")
async def live_forecast(date: str = None, model: str = "Ensemble"):
    """
    Prediction endpoint handler for real-time live forecasting.
    
    Inputs:
      - date: Optional query parameter string (YYYY-MM-DD).
      - model: Target model name string (default: "Ensemble").
      
    Outputs:
      - JSON containing dynamically scraped features and computed 24-step predictions.
      
    ML Principle: Real-time feature engineering and dynamic HTTP endpoint data serialization.
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        if date is None:
            date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        result = model_manager.run_dual_forecast(date, selected_model=model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize")
async def optimize(request: OptimizationRequest):
    """
    Endpoint for activity scheduling using Linear Programming based on ML price forecasts.
    
    Inputs:
      - request: OptimizationRequest JSON containing list of ActivityInput constraints.
      
    Outputs:
      - JSON containing hourly schedule mappings minimizing total expected cost.
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        date_str = request.date
        if not date_str:
            date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Get forecast (uses selected model, defaults to Ensemble)
        selected_model = request.model or "Ensemble"
        forecast = model_manager.run_dual_forecast(date_str, selected_model=selected_model)
        price_forecast = forecast["predicted_price"]
        consumption_forecast = forecast["predicted_consumption"]

        activities = [
            Activity(
                name=a.name, power_kw=a.power_kw,
                duration_hours=a.duration_hours, priority=a.priority,
                earliest_start=a.earliest_start, latest_finish=a.latest_finish,
                icon=a.icon
            )
            for a in request.activities
        ]
        plan = generate_optimal_plan(activities, price_forecast, consumption_forecast)
        return {
            "schedule": [
                {"name": s.name, "start_hour": s.start_hour, "end_hour": s.end_hour,
                 "power_kw": s.power_kw, "duration_hours": s.duration_hours,
                 "priority": s.priority, "cost_optimized": s.cost_optimized,
                 "cost_baseline": s.cost_baseline, "savings": s.savings, "icon": s.icon}
                for s in plan.schedule
            ],
            "hourly_power_profile": plan.hourly_power_profile,
            "hourly_price": plan.hourly_price,
            "hourly_consumption_forecast": plan.hourly_consumption_forecast,
            "total_cost_optimized": plan.total_cost_optimized,
            "total_cost_baseline": plan.total_cost_baseline,
            "total_savings": plan.total_savings,
            "savings_percentage": plan.savings_percentage,
            "forecasted_minimum_bill": plan.forecasted_minimum_bill,
            "cheapest_hour": plan.cheapest_hour,
            "most_expensive_hour": plan.most_expensive_hour,
            "solver_status": plan.solver_status,
            "date": date_str,
            "model_used": request.model or "Ensemble",
            "data_source": forecast.get("data_source", "Unknown"),
            "ensemble_config": forecast.get("ensemble_config", {}),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/default-activities")
async def get_default_activities():
    """Return default household activity presets."""
    return {"activities": DEFAULT_ACTIVITIES}


# ═════════════════════════════════════════════════════════════
# RESULTS & VALIDATION ENDPOINTS
# ═════════════════════════════════════════════════════════════

@app.get("/api/validation-results")
async def get_validation_results():
    """Return validation metrics from validation/metrics_summary.csv."""
    csv_path = PROJECT_ROOT / "validation" / "metrics_summary.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Validation metrics file not found")
    try:
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "model": row.get("Model", ""),
                    "target": row.get("Target", ""),
                    "rmse": float(row.get("RMSE", 0)),
                    "mae": float(row.get("MAE", 0)),
                    "mape": float(row.get("MAPE", 0)),
                    "r2": float(row.get("R2", 0)),
                    "samples": int(row.get("N_Samples", 0)),
                })
        return {"phase": "validation", "period": "2025 (8,730 samples)", "metrics": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-results")
async def get_test_results():
    """Return test metrics from test_results/test_metrics_summary.csv."""
    csv_path = PROJECT_ROOT / "test_results" / "test_metrics_summary.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Test metrics file not found")
    try:
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "model": row.get("Model", ""),
                    "target": row.get("Type", ""),
                    "rmse": float(row.get("RMSE", 0)),
                    "mae": float(row.get("MAE", 0)),
                    "mape": float(row.get("MAPE", 0)),
                    "r2": float(row.get("R2", 0)),
                    "samples": int(row.get("Samples", 0)),
                })
        return {"phase": "test", "period": "Jan–Apr 2026 (2,364 samples)", "metrics": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results-plots/{phase}/{filename}")
async def get_results_plot(phase: str, filename: str):
    """Serve pre-generated plot images from validation/ or test_results/."""
    if phase not in ("validation", "test_results"):
        raise HTTPException(status_code=400, detail="Phase must be 'validation' or 'test_results'")
    file_path = PROJECT_ROOT / phase / filename
    if not file_path.exists() or not filename.endswith(".png"):
        raise HTTPException(status_code=404, detail=f"Plot not found: {phase}/{filename}")
    return FileResponse(str(file_path), media_type="image/png")


@app.get("/api/available-plots")
async def get_available_plots():
    """List all available plot images from validation/ and test_results/."""
    plots = {"validation": [], "test_results": []}
    for phase in ["validation", "test_results"]:
        phase_dir = PROJECT_ROOT / phase
        if phase_dir.exists():
            for f in sorted(phase_dir.glob("*.png")):
                plots[phase].append(f.name)
    return plots


@app.get("/api/feature-importances")
async def get_feature_importances():
    """Return XGBoost feature importances for consumption and price models."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        models = model_manager.models
        feature_cols = models['feature_cols']
        result = {}

        # XGBoost feature importance (gain)
        xgb_cons = models['xgb_cons']
        xgb_price = models['xgb_price']

        # Get importance for first estimator (h+1 target) since models are wrapped in MultiOutputRegressor
        cons_imp = xgb_cons.estimators_[0].feature_importances_ if hasattr(xgb_cons, 'estimators_') else xgb_cons.feature_importances_
        price_imp = xgb_price.estimators_[0].feature_importances_ if hasattr(xgb_price, 'estimators_') else xgb_price.feature_importances_

        # Build top-15 features
        cons_top_idx = cons_imp.argsort()[-15:][::-1]
        price_top_idx = price_imp.argsort()[-15:][::-1]

        result["XGBoost"] = {
            "consumption": [
                {"name": feature_cols[i], "importance": float(cons_imp[i])}
                for i in cons_top_idx
            ],
            "price": [
                {"name": feature_cols[i], "importance": float(price_imp[i])}
                for i in price_top_idx
            ],
        }

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "ML Energy Prediction & Optimization API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "available_dates": "GET /available-dates",
            "predict": "POST /api/predict",
            "live_forecast": "GET /api/live-forecast?date=YYYY-MM-DD",
            "optimize": "POST /api/optimize",
            "default_activities": "GET /api/default-activities",
            "validation_results": "GET /api/validation-results",
            "test_results": "GET /api/test-results",
            "feature_importances": "GET /api/feature-importances",
            "available_plots": "GET /api/available-plots",
            "results_plot": "GET /api/results-plots/{phase}/{filename}",
        }
    }


# ─────────────────────────────────────────
# Run with: cd backend && uvicorn api:app --reload --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

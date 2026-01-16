from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import json
from typing import Optional, Dict, Any, List, Literal
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="BioOptima: Penicillin Process Analytics", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
data_store = {
    "raw_data": None,
    "preprocessed_data": None,
    "models": {
        "linear": None,
        "forest": None
    },
    "scaler": None,
    "feature_scaler": None, 
    "model_metadata": None,
    "feature_names": None,
    "target_name": None,
    "column_mapping": {}
}

# Key Process Parameters
KEY_PARAMETERS = {
    "sugar_feed": ["sugar", "feed", "fs", "glucose"],
    "paa": ["paa", "phenylacetic", "fpaa"],
    "agitator_rpm": ["rpm", "agitator", "agitation"],
    "aeration_rate": ["aeration", "air", "fg", "o2"]
}

# ============================================
# HELPER FUNCTIONS
# ============================================
def clean_nans(data):
    """Recursively replace NaN and Infinity with None for JSON compliance"""
    if isinstance(data, dict):
        return {k: clean_nans(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_nans(v) for v in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
        return data
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    return data

def find_column(df_cols, keywords):
    for col in df_cols:
        col_lower = col.lower()
        for key in keywords:
            if key in col_lower:
                return col
    return None

def identify_biotech_columns(df):
    mapping = {}
    for key, keywords in KEY_PARAMETERS.items():
        found_col = find_column(df.columns, keywords)
        if found_col:
            mapping[key] = found_col
    mapping["time"] = find_column(df.columns, ["time", "hour", "duration"])
    mapping["batch_id"] = find_column(df.columns, ["batch", "id", "run"])
    mapping["target"] = find_column(df.columns, ["penicillin", "product", "concentration", "yield", "output"])
    return mapping

# ============================================
# API 1: UPLOAD DATA (ENHANCED)
# ============================================
@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Uploads data and provides deep batch comparisons (Golden vs Bad Batch).
    """
    try:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), header=1)
            if not any("batch" in col.lower() for col in df.columns):
                 df = pd.read_csv(io.BytesIO(contents), header=0)
        except:
             df = pd.read_csv(io.BytesIO(contents), header=0)

        data_store["raw_data"] = df
        mapping = identify_biotech_columns(df)
        data_store["column_mapping"] = mapping
        
        target_col = mapping.get("target")
        batch_col = mapping.get("batch_id")
        time_col = mapping.get("time")

        # --- Insight 1: Batch-Level Yield Analysis ---
        batch_insights = {}
        visualization_data = {}
        
        if batch_col and target_col:
            batch_groups = df.groupby(batch_col)
            batch_yields = batch_groups[target_col].max()
            
            # Identify Extremes
            best_batch_id = batch_yields.idxmax()
            worst_batch_id = batch_yields.idxmin()
            
            batch_insights = {
                "total_batches": int(df[batch_col].nunique()),
                "average_yield": round(float(batch_yields.mean()), 2),
                "max_yield": round(float(batch_yields.max()), 2),
                "min_yield": round(float(batch_yields.min()), 2),
                "golden_batch_id": int(best_batch_id),
                "lowest_yield_batch_id": int(worst_batch_id),
                "yield_std_dev": round(float(batch_yields.std()), 2)
            }
            
            # --- Graph Data: Raw Traces (Spaghetti Plot) ---
            # Extract time-series for first 5 batches to show variability
            raw_traces = []
            if time_col:
                for bid in df[batch_col].unique()[:5]: # Limit to 5 batches
                    batch_data = df[df[batch_col] == bid]
                    # Downsample for performance (every 10th point)
                    batch_data = batch_data.iloc[::10, :]
                    raw_traces.append({
                        "batch_id": int(bid),
                        "time": batch_data[time_col].tolist(),
                        "yield": batch_data[target_col].tolist()
                    })
            visualization_data["raw_traces"] = raw_traces

            # --- Graph Data: Yield Distribution ---
            clean_yields = batch_yields.dropna()
            if not clean_yields.empty:
                hist_counts, hist_bins = np.histogram(clean_yields, bins=15)
                visualization_data["yield_distribution"] = {
                    "x_values": [round(b, 2) for b in hist_bins[:-1]],
                    "y_values": hist_counts.tolist()
                }

        # --- Insight 2: Feature Ranges (Data Quality) ---
        feature_ranges = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for clean_name, original_name in mapping.items():
            if clean_name in KEY_PARAMETERS and original_name in numeric_cols:
                feature_ranges.append({
                    "parameter": clean_name.replace("_", " ").title(),
                    "min": float(df[original_name].min()),
                    "max": float(df[original_name].max()),
                    "mean": float(df[original_name].mean())
                })

        response = {
            "success": True,
            "filename": file.filename,
            "biotech_insights": {
                "batch_summary": batch_insights,
                "parameter_ranges": feature_ranges,
            },
            "dashboard_graphs": visualization_data
        }
        return clean_nans(response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing file: {str(e)}")


# ============================================
# API 2: PREPROCESS (ENHANCED STATS)
# ============================================
@app.post("/api/preprocess")
async def preprocess_data():
    """
    Cleans data and calculates advanced statistical moments (Skew/Kurtosis).
    """
    try:
        if data_store["raw_data"] is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        df = data_store["raw_data"].copy()
        mapping = data_store["column_mapping"]
        
        # 1. Imputation
        missing_count = int(df.isnull().sum().sum())
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 2. Advanced Stats (Skewness & Kurtosis)
        # These indicate if sensors are noisy or if process drifts
        stats_data = []
        for clean_name, original_name in mapping.items():
            if clean_name in KEY_PARAMETERS:
                series = df[original_name]
                stats_data.append({
                    "parameter": clean_name.replace("_", " ").title(),
                    "skewness": float(skew(series)),
                    "kurtosis": float(kurtosis(series)),
                    "status": "Stable" if abs(skew(series)) < 1 else "Drifting/Skewed"
                })

        # 3. Trajectories (Golden Batch)
        time_col = mapping.get("time")
        target_col = mapping.get("target")
        trajectory_data = []
        
        if time_col and target_col:
            df['approx_time'] = (df[time_col] * 2).round() / 2
            grouped = df.groupby('approx_time')
            cols_to_track = [target_col] + [v for k, v in mapping.items() if k in KEY_PARAMETERS]
            
            for t, group in grouped:
                if t > 60: continue # Cap at 60h
                point = {"time": float(t)}
                for col in cols_to_track:
                    if col:
                        point[f"{col}_mean"] = float(group[col].mean())
                        point[f"{col}_std"] = float(group[col].std())
                trajectory_data.append(point)

        data_store["preprocessed_data"] = df
        
        return clean_nans({
            "success": True,
            "stats_deep_dive": {
                "imputation_count": missing_count,
                "sensor_health": stats_data
            },
            "visualization_data": {
                "golden_batch_trajectory": trajectory_data
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# API 3: TRAIN (LINEAR VS FOREST)
# ============================================
class TrainingConfig(BaseModel):
    test_size: float = 0.2
    n_estimators: int = 100 # For Random Forest

@app.post("/api/train")
async def train_models(config: TrainingConfig):
    """
    Trains BOTH Linear Regression and Random Forest for comparison.
    """
    try:
        df = data_store["preprocessed_data"]
        mapping = data_store["column_mapping"]
        
        feature_cols = [mapping[k] for k in KEY_PARAMETERS if k in mapping]
        target_col = mapping.get("target")
        
        if not feature_cols or not target_col:
            raise HTTPException(status_code=400, detail="Key columns missing")

        X = df[feature_cols]
        y = df[target_col]
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=config.test_size, random_state=42
        )
        
        # --- MODEL 1: LINEAR REGRESSION ---
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        lr_metrics = {
            "r2": r2_score(y_test, lr_pred),
            "mae": mean_absolute_error(y_test, lr_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, lr_pred))
        }

        # --- MODEL 2: RANDOM FOREST ---
        rf_model = RandomForestRegressor(n_estimators=config.n_estimators, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        rf_metrics = {
            "r2": r2_score(y_test, rf_pred),
            "mae": mean_absolute_error(y_test, rf_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, rf_pred))
        }

        # Store Artifacts
        data_store["models"]["linear"] = lr_model
        data_store["models"]["forest"] = rf_model
        data_store["scaler"] = scaler
        data_store["feature_names"] = feature_cols
        data_store["target_name"] = target_col
        
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(df[feature_cols])
        data_store["feature_scaler"] = feature_scaler

        # Feature Importance Comparison
        # Linear uses coefficients, RF uses feature_importances_
        feature_impact = []
        for i, col_name in enumerate(feature_cols):
            clean_name = next(k for k, v in mapping.items() if v == col_name)
            feature_impact.append({
                "parameter": clean_name.replace("_", " ").title(),
                "linear_coef": float(lr_model.coef_[i]),
                "forest_importance": float(rf_model.feature_importances_[i])
            })

        return clean_nans({
            "success": True,
            "comparison": {
                "linear": lr_metrics,
                "forest": rf_metrics,
                "winner": "Random Forest" if rf_metrics['r2'] > lr_metrics['r2'] else "Linear Regression"
            },
            "feature_analysis": feature_impact,
            "visualization_data": {
                "actual_vs_predicted": {
                    "actual": y_test.tolist()[:100], # Limit points
                    "linear_pred": lr_pred.tolist()[:100],
                    "forest_pred": rf_pred.tolist()[:100]
                }
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# API 4: PREDICT (MULTI-MODEL)
# ============================================
class ProcessParams(BaseModel):
    sugar_feed: float
    paa: float
    agitator_rpm: float
    aeration_rate: float
    model_type: Literal["linear", "forest"] = "forest" # User choice

@app.post("/api/predict")
async def predict_process(params: ProcessParams):
    """
    Predicts using the selected model type.
    """
    try:
        model = data_store["models"].get(params.model_type)
        if not model:
            raise HTTPException(status_code=400, detail=f"Model {params.model_type} not trained")
            
        mapping = data_store["column_mapping"]
        feature_names = data_store["feature_names"]
        
        # 1. Prepare Input
        input_values = []
        for col in feature_names:
            key_match = next(k for k, v in mapping.items() if v == col)
            val = getattr(params, key_match)
            input_values.append(val)
            
        input_array = np.array([input_values])
        input_scaled = data_store["scaler"].transform(input_array)
        
        # 2. Predict
        prediction = model.predict(input_scaled)[0]
        
        # 3. Contextual Analysis (Radar)
        norm_inputs = data_store["feature_scaler"].transform(input_array)[0]
        radar_data = []
        for i, col in enumerate(feature_names):
            key_match = next(k for k, v in mapping.items() if v == col)
            radar_data.append({
                "parameter": key_match.replace("_", " ").title(),
                "value": float(input_values[i]),
                "percentile": float(norm_inputs[i] * 100)
            })
            
        return clean_nans({
            "prediction": {
                "value": round(float(prediction), 4),
                "model_used": params.model_type.title()
            },
            "visualization_data": {
                "radar_indicators": radar_data
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
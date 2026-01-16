from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import json
from typing import Optional, Dict, Any, List
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="BioOptima: Penicillin Process Analytics", version="3.0.0")

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
    "trained_model": None,
    "scaler": None,
    "feature_scaler": None, # For radar charts
    "model_metadata": None,
    "feature_names": None,
    "target_name": None,
    "column_mapping": {} # Maps simplified names to CSV columns
}

# Key Process Parameters (Your "Hyperparameters")
KEY_PARAMETERS = {
    "sugar_feed": ["sugar", "feed", "fs", "glucose"],
    "paa": ["paa", "phenylacetic", "fpaa"],
    "agitator_rpm": ["rpm", "agitator", "agitation"],
    "aeration_rate": ["aeration", "air", "fg", "o2"]
}

# ============================================
# HELPER FUNCTIONS
# ============================================
def find_column(df_cols, keywords):
    """Fuzzy search for column names"""
    for col in df_cols:
        col_lower = col.lower()
        for key in keywords:
            if key in col_lower:
                return col
    return None

def identify_biotech_columns(df):
    """Map the messy CSV columns to clean internal names"""
    mapping = {}
    
    # Map Key Parameters
    for key, keywords in KEY_PARAMETERS.items():
        found_col = find_column(df.columns, keywords)
        if found_col:
            mapping[key] = found_col
            
    # Map Context Parameters
    mapping["time"] = find_column(df.columns, ["time", "hour", "duration"])
    mapping["batch_id"] = find_column(df.columns, ["batch", "id", "run"])
    mapping["target"] = find_column(df.columns, ["penicillin", "product", "concentration", "yield", "output"])
    
    return mapping

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
    elif isinstance(data, np.integer): # Handle numpy integers
        return int(data)
    elif isinstance(data, np.floating): # Handle numpy floats
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    return data

# ============================================
# API 1: UPLOAD DATA (FIXED)
# ============================================
@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Uploads fermentation data and performs Batch-Level Analysis.
    """
    try:
        contents = await file.read()
        # Try reading with header=1 first as biotech CSVs often have units in row 2
        try:
            df = pd.read_csv(io.BytesIO(contents), header=1)
            # Basic validation: check if 'Batch' or 'Time' exists, if not retry with header=0
            if not any("batch" in col.lower() for col in df.columns):
                 df = pd.read_csv(io.BytesIO(contents), header=0)
        except:
             df = pd.read_csv(io.BytesIO(contents), header=0)

        data_store["raw_data"] = df
        
        # Identify Columns
        mapping = identify_biotech_columns(df)
        data_store["column_mapping"] = mapping
        
        target_col = mapping.get("target")
        batch_col = mapping.get("batch_id")
        time_col = mapping.get("time")

        # --- Insight 1: Batch-Level Yield Analysis ---
        batch_insights = {}
        visualization_data = {}
        
        if batch_col and target_col:
            # Group by Batch to find final yield per batch
            batch_groups = df.groupby(batch_col)
            batch_yields = batch_groups[target_col].max()
            
            # Identify "Golden Batch" (Best Yield)
            best_batch_id = batch_yields.idxmax()
            best_batch_yield = batch_yields.max()
            avg_yield = batch_yields.mean()
            std_yield = batch_yields.std()
            
            # Safe CV calculation
            cv = 0
            if avg_yield > 0 and not np.isnan(std_yield):
                cv = (std_yield / avg_yield * 100)
            
            batch_insights = {
                "total_batches": int(df[batch_col].nunique()),
                "average_yield": round(float(avg_yield), 2),
                "max_yield": round(float(best_batch_yield), 2),
                "golden_batch_id": int(best_batch_id),
                "yield_variability_cv": round(float(cv), 2)
            }
            
            # Histogram Data for Yield Distribution
            # Dropna ensures np.histogram doesn't get NaNs
            clean_yields = batch_yields.dropna()
            if not clean_yields.empty:
                hist_counts, hist_bins = np.histogram(clean_yields, bins=10)
                visualization_data["yield_distribution"] = {
                    "type": "bar",
                    "title": "Batch Yield Distribution",
                    "x_label": "Penicillin Concentration (g/L)",
                    "y_label": "Number of Batches",
                    "x_values": [round(b, 2) for b in hist_bins[:-1]],
                    "y_values": hist_counts.tolist()
                }

        # --- Insight 2: Input Parameter Correlation ---
        correlations = []
        if target_col:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Fill NaNs before correlation to avoid errors
            corr_matrix = df[numeric_cols].fillna(0).corr()
            
            if target_col in corr_matrix:
                target_corrs = corr_matrix[target_col].sort_values(ascending=False)
                
                # Filter for our key parameters
                for clean_name, original_name in mapping.items():
                    if clean_name in KEY_PARAMETERS and original_name in target_corrs:
                        val = target_corrs[original_name]
                        # Check for NaN correlations (happens if column is constant)
                        if not np.isnan(val):
                            correlations.append({
                                "parameter": clean_name.replace("_", " ").title(),
                                "correlation": round(float(val), 3),
                                "relationship": "Positive" if val > 0 else "Negative"
                            })

        response_data = {
            "success": True,
            "filename": file.filename,
            "biotech_insights": {
                "batch_summary": batch_insights,
                "key_parameter_correlations": correlations,
                "data_health": {
                    "missing_values": int(df.isnull().sum().sum()),
                    "columns_mapped": [k for k, v in mapping.items() if v is not None]
                }
            },
            "dashboard_graphs": visualization_data
        }
        
        # Sanitize entire response before returning
        return clean_nans(response_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error to console for debugging
        raise HTTPException(status_code=400, detail=f"Error analyzing file: {str(e)}")

# ============================================
# API 2: PREPROCESS & TRAJECTORY ANALYSIS
# ============================================
@app.post("/api/preprocess")
async def preprocess_data():
    """
    Prepares data and calculates 'Golden Batch' trajectories.
    """
    try:
        if data_store["raw_data"] is None:
            raise HTTPException(status_code=400, detail="No data uploaded")
        
        df = data_store["raw_data"].copy()
        mapping = data_store["column_mapping"]
        
        # 1. Clean Data (Impute & Drop)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 2. Generate "Standard Trajectory" (Avg Process Profile)
        # We group by rounded time to average across batches
        time_col = mapping.get("time")
        target_col = mapping.get("target")
        trajectories = {}
        
        if time_col:
            # Round time to nearest 0.5h for grouping
            df['approx_time'] = (df[time_col] * 2).round() / 2
            grouped = df.groupby('approx_time')
            
            # Calculate Mean and Std for Target and Key Parameters
            cols_to_track = [target_col] + [v for k, v in mapping.items() if k in KEY_PARAMETERS]
            
            trajectory_data = []
            for t, group in grouped:
                if t > 50: continue # Limit to first 50h for frontend performance or sample
                point = {"time": float(t)}
                for col in cols_to_track:
                    if col:
                        point[f"{col}_mean"] = float(group[col].mean())
                        point[f"{col}_std"] = float(group[col].std())
                trajectory_data.append(point)
            
            trajectories["process_profiles"] = trajectory_data

        data_store["preprocessed_data"] = df
        
        return {
            "success": True,
            "message": "Data cleaned and profiles generated",
            "preprocessing_insights": {
                "samples_processed": len(df),
                "features_tracked": list(KEY_PARAMETERS.keys())
            },
            "visualization_data": {
                "golden_batch_trajectory": {
                    "type": "line_with_confidence",
                    "title": "Average Fermentation Profile (Mean ± Std Dev)",
                    "data": trajectory_data  # Frontend: Plot Mean as line, Mean+/-Std as shaded area
                }
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# API 3: TRAIN MODEL
# ============================================
class TrainingConfig(BaseModel):
    test_size: float = 0.2

@app.post("/api/train")
async def train_model(config: TrainingConfig):
    """
    Trains Linear Regression specifically on the identified Critical Process Parameters.
    """
    try:
        df = data_store["preprocessed_data"]
        mapping = data_store["column_mapping"]
        
        # Select Features: Only use the Key Parameters identified
        feature_cols = [mapping[k] for k in KEY_PARAMETERS if k in mapping]
        target_col = mapping.get("target")
        
        if not feature_cols or not target_col:
            raise HTTPException(status_code=400, detail="Key biotech columns not found in data")

        X = df[feature_cols]
        y = df[target_col]
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=config.test_size)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Metrics
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Save artifacts
        data_store["trained_model"] = model
        data_store["scaler"] = scaler
        data_store["feature_names"] = feature_cols
        data_store["target_name"] = target_col
        
        # Save a MinMax scaler for Radar Charts later
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(df[feature_cols])
        data_store["feature_scaler"] = feature_scaler

        # Feature Importance (Coefficient Analysis)
        importance = []
        for name, col_name in mapping.items():
            if col_name in feature_cols:
                idx = feature_cols.index(col_name)
                coef = model.coef_[idx]
                importance.append({
                    "parameter": name.replace("_", " ").title(),
                    "impact_score": float(coef),
                    "interpretation": "Increases Yield" if coef > 0 else "Decreases Yield"
                })

        return {
            "success": True,
            "model_performance": {
                "r2_score": round(r2, 4),
                "model_type": "Linear Regression (Process Parameters Only)"
            },
            "biotech_insights": {
                "key_drivers": sorted(importance, key=lambda x: abs(x['impact_score']), reverse=True),
                "model_equation": "Yield = " + " + ".join([f"({i['impact_score']:.2f} * {i['parameter']})" for i in importance])
            },
            "visualization_data": {
                "actual_vs_predicted": {
                    "actual": y_test.tolist()[:50],
                    "predicted": y_pred.tolist()[:50]
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# API 4: PREDICT & OPTIMIZE
# ============================================
class ProcessParams(BaseModel):
    sugar_feed: float
    paa: float
    agitator_rpm: float
    aeration_rate: float

@app.post("/api/predict")
async def predict_process(params: ProcessParams):
    """
    Predicts yield and visualizes where the input parameters sit in the 'Design Space'.
    """
    try:
        if not data_store["trained_model"]:
            raise HTTPException(status_code=400, detail="Model not trained")
            
        mapping = data_store["column_mapping"]
        feature_names = data_store["feature_names"]
        
        # 1. Prepare Input
        # Map user input to the correct column order used in training
        input_values = []
        for col in feature_names:
            # Find which key matches this column
            key_match = next(k for k, v in mapping.items() if v == col)
            val = getattr(params, key_match)
            input_values.append(val)
            
        input_array = np.array([input_values])
        
        # 2. Predict
        input_scaled = data_store["scaler"].transform(input_array)
        prediction = data_store["trained_model"].predict(input_scaled)[0]
        
        # 3. Contextual Analysis (Radar Chart Data)
        # Normalize inputs to 0-100% relative to historical data range
        # This shows if a parameter is High, Low, or Medium compared to past batches
        norm_inputs = data_store["feature_scaler"].transform(input_array)[0]
        
        radar_data = []
        for i, col in enumerate(feature_names):
            key_match = next(k for k, v in mapping.items() if v == col)
            radar_data.append({
                "parameter": key_match.replace("_", " ").title(),
                "value": float(input_values[i]),
                "percentile_score": float(norm_inputs[i] * 100), # 0 = min historical, 100 = max historical
                "status": "High" if norm_inputs[i] > 0.8 else "Low" if norm_inputs[i] < 0.2 else "Normal"
            })
            
        # 4. Benchmarking
        # Compare prediction to historical max yield
        raw_df = data_store["raw_data"]
        target_col = data_store["target_name"]
        max_yield = raw_df[target_col].max()
        avg_yield = raw_df[target_col].mean()
        
        return {
            "prediction": {
                "predicted_penicillin_conc": round(float(prediction), 4),
                "unit": "g/L",
                "performance_assessment": "Excellent" if prediction > avg_yield * 1.1 else "Average" if prediction > avg_yield * 0.9 else "Poor"
            },
            "optimization_insights": {
                "comparison_to_max": f"{round(prediction/max_yield*100, 1)}% of Best Historical Batch",
                "parameter_health_check": [
                    f"{r['parameter']} is {r['status']} ({r['percentile_score']:.0f}% of range)" 
                    for r in radar_data if r['status'] != "Normal"
                ]
            },
            "visualization_data": {
                "radar_chart": {
                    "title": "Parameter Design Space",
                    "indicators": radar_data  # Frontend: Plot these on a spider/radar chart
                },
                "yield_gauge": {
                    "value": round(float(prediction), 2),
                    "min": float(raw_df[target_col].min()),
                    "mean": float(avg_yield),
                    "max": float(max_yield)
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
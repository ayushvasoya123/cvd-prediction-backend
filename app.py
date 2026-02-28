from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "data", "cardio_train_Dataset.csv")
METRICS_PATH = os.path.join(BASE_DIR, "metrics", "model_metrics.json")
VIZ_PATH = os.path.join(BASE_DIR, "metrics", "visualization.json")

FEATURE_NAMES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "bmi"
]
DATASET_VIEW_COLUMNS = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "bmi", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
]

_DATASET_CACHE = None
_DATASET_CACHE_MTIME = None

model = pickle.load(open("models/model.pkl", "rb"))


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_dataset():
    global _DATASET_CACHE, _DATASET_CACHE_MTIME
    if not os.path.exists(DATASET_PATH):
        return None

    mtime = os.path.getmtime(DATASET_PATH)
    if _DATASET_CACHE is not None and _DATASET_CACHE_MTIME == mtime:
        return _DATASET_CACHE.copy()

    # Auto-detect CSV delimiter (dataset commonly ships with ';' separator).
    df = pd.read_csv(DATASET_PATH, sep=None, engine="python")

    if "age" in df.columns:
        df["age"] = (pd.to_numeric(df["age"], errors="coerce") / 365.25).round().astype("Int64")

    for col in ["height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio", "gender"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"weight", "height"}.issubset(df.columns):
        height_m = df["height"] / 100
        bmi = df["weight"] / (height_m * height_m)
        df["bmi"] = bmi.round(2)
    else:
        df["bmi"] = np.nan

    for col in DATASET_VIEW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[DATASET_VIEW_COLUMNS].dropna(subset=["age", "cardio"]).copy()
    _DATASET_CACHE = df.copy()
    _DATASET_CACHE_MTIME = mtime
    return df


def _apply_dataset_filters(df, cardio_filter, high_bp_only):
    filtered = df
    if cardio_filter in ("0", "1"):
        filtered = filtered[filtered["cardio"] == int(cardio_filter)]
    if high_bp_only:
        filtered = filtered[filtered["ap_hi"] > 140]
    return filtered


def _serialize_dataset_rows(frame):
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "age": int(row["age"]) if pd.notna(row["age"]) else None,
            "gender": int(row["gender"]) if pd.notna(row["gender"]) else None,
            "height": float(row["height"]) if pd.notna(row["height"]) else None,
            "weight": float(row["weight"]) if pd.notna(row["weight"]) else None,
            "ap_hi": int(row["ap_hi"]) if pd.notna(row["ap_hi"]) else None,
            "ap_lo": int(row["ap_lo"]) if pd.notna(row["ap_lo"]) else None,
            "bmi": float(row["bmi"]) if pd.notna(row["bmi"]) else None,
            "cholesterol": int(row["cholesterol"]) if pd.notna(row["cholesterol"]) else None,
            "gluc": int(row["gluc"]) if pd.notna(row["gluc"]) else None,
            "smoke": int(row["smoke"]) if pd.notna(row["smoke"]) else None,
            "alco": int(row["alco"]) if pd.notna(row["alco"]) else None,
            "active": int(row["active"]) if pd.notna(row["active"]) else None,
            "cardio": int(row["cardio"]) if pd.notna(row["cardio"]) else None,
        })
    return rows


@app.route("/", methods=["GET"])
def home():
    return "<h1 style='color:green; text-align:center;'>Cardio AI Backend is Online</h1>"


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return _build_cors_prelight_response()

    try:
        data = request.json
        if not data or "features" not in data:
            return jsonify({"error": "No features provided"}), 400

        features_df = pd.DataFrame([data["features"]], columns=FEATURE_NAMES)
        prob = model.predict_proba(features_df)[0][1]
        pred = int(prob >= 0.5)

        return jsonify({
            "prediction": pred,
            "probability": float(prob)
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found"}), 404


@app.route("/visualization", methods=["GET"])
def get_visual_file():
    try:
        with open(VIZ_PATH, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "visualization.json not found. Run the script first."}), 404


@app.route("/dataset", methods=["GET"])
def dataset():
    df = _load_dataset()
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404

    page = max(1, _safe_int(request.args.get("page"), 1))
    page_size = min(100, max(5, _safe_int(request.args.get("page_size"), 10)))
    cardio_filter = request.args.get("cardio", "")
    high_bp_only = request.args.get("high_bp", "false").lower() == "true"

    filtered = _apply_dataset_filters(df, cardio_filter, high_bp_only)
    filtered_records = int(len(filtered))
    total_pages = max(1, int(np.ceil(filtered_records / page_size))) if filtered_records else 1
    page = min(page, total_pages)

    start = (page - 1) * page_size
    end = start + page_size
    page_frame = filtered.iloc[start:end]

    total_records = int(len(df))
    disease_records = int((df["cardio"] == 1).sum())
    disease_prevalence = round((disease_records / total_records) * 100, 2) if total_records else 0.0
    avg_age = round(float(df["age"].mean()), 1) if total_records else 0.0

    return jsonify({
        "rows": _serialize_dataset_rows(page_frame),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "filtered_records": filtered_records,
            "start_index": start + 1 if filtered_records else 0,
            "end_index": min(end, filtered_records),
        },
        "filters": {
            "cardio": cardio_filter,
            "high_bp": high_bp_only,
        },
        "stats": {
            "total_records": total_records,
            "disease_records": disease_records,
            "disease_prevalence_pct": disease_prevalence,
            "avg_age_years": avg_age,
        },
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/dataset/export", methods=["GET"])
def dataset_export():
    df = _load_dataset()
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404

    cardio_filter = request.args.get("cardio", "")
    high_bp_only = request.args.get("high_bp", "false").lower() == "true"
    filtered = _apply_dataset_filters(df, cardio_filter, high_bp_only)

    csv_data = filtered.to_csv(index=False)
    response = make_response(csv_data)
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=filtered_dataset.csv"
    return response


@app.route("/dashboard", methods=["GET"])
def dashboard():
    df = _load_dataset()
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404

    total_records = int(len(df))
    disease_records = int((df["cardio"] == 1).sum())
    disease_prevalence = round((disease_records / total_records) * 100, 2) if total_records else 0.0
    high_bp_records = int((df["ap_hi"] > 140).sum())
    avg_age_years = round(float(df["age"].mean()), 1) if total_records else None

    metrics_data = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

    response = {
        "dataset": {
            "total_records": total_records,
            "disease_records": disease_records,
            "disease_prevalence_pct": disease_prevalence,
            "high_bp_records": high_bp_records,
            "feature_count": len(FEATURE_NAMES),
            "avg_age_years": avg_age_years,
        },
        "model": {
            "name": "Random Forest",
            "algorithms_tested": 5,
            "metrics": metrics_data,
        },
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(response)


def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5000)

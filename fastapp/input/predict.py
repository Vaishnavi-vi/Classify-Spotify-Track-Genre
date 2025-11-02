
import pandas as pd
import pickle
from fastapp.input import user_input
import mlflow.xgboost
import logging
import os
import numpy as np


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("genre_prediction")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "genre_prediction.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------- Load Artifacts --------------------
try:
    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("artifacts/power_transformer.pkl", "rb") as f:
        pt = pickle.load(f)

    logger.debug("Scaler and Preprocess loaded successfully!!")

except Exception as e:
    logger.error("Error loading artifacts: %s", e)
    raise


try:
    model = mlflow.xgboost.load_model("models:/xgboost_classifier/1")
    logger.debug("Model loaded successfully from MLflow registry.")
except Exception as e:
    logger.warning("MLflow model not available (%s). Loading from pickle...", e)
    try:
        with open("artifacts/xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully from pickle file.")
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise



def prediction(user_input: dict):
    try:
        # Convert input dict to DataFrame
        df = pd.DataFrame([user_input])

        # Ensure all expected columns are present
        skewed_col = ["duration_ms", "loudness", "speechiness", "instrumentalness", "liveness", "time_signature"]
        df[skewed_col] = pt.transform(df[skewed_col])

        # Apply preprocessing (PowerTransformer)
        
        df.drop(["key","explicit"],axis=1,inplace=True)
        # Apply StandardScaler
        
        scaled_df=scaler.transform(df)

        # Predict genre
        prediction_result = model.predict(scaled_df)[0]
        probabilities = model.predict_proba(scaled_df)[0]
        confidence = float(np.max(probabilities))

        # Map cluster ID to readable genre
        cluster_map = {
            0: "Organic/Classic Vibes",
            1: "Modern & Energetic Beats",
            2: "World & Mood Music"
        }
        predicted_label = cluster_map.get(prediction_result, "Unknown Genre")

        logger.debug("Prediction completed successfully.")
        return {
            "prediction_category": predicted_label,
            "confidence": confidence,
            "class_probabilities": [float(p) for p in probabilities]
        }

    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise

    


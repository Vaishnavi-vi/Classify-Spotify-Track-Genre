import pandas as pd
import pickle
import mlflow
import mlflow.xgboost
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score, recall_score, confusion_matrix,f1_score
import json

# ------------------- Logging Setup -------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join(log_dir, "model_evaluation.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def evaluate_model(model_path: str, x_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Evaluate trained model, log metrics and confusion matrix to MLflow.
    """
    try:
        logger.info("Starting model evaluation...")

        with mlflow.start_run(run_name="XGBoost_Evaluation"):

            # Load model
            with open(model_path,"rb") as f:
                model=pickle.load(f)
            logger.info("Model loaded successfully.")

            # Predictions
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)


            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,

            }
            
            os.makedirs("artifacts",exist_ok=True)
            with open("artifacts/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            logger.info("Metrics logged to MLflow.")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()

            # Save confusion matrix
            os.makedirs("artifacts", exist_ok=True)
            plot_path = "artifacts/confusion_matrix.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            logger.info("Confusion matrix saved and logged as artifact.")

            # Log model for version tracking
            mlflow.xgboost.log_model(model, "evaluated_model")

            logger.info(f"Evaluation done | Accuracy: {acc:.4f}, F1: {f1:.4f}")
            return metrics

    except Exception as e:
        logger.error(" Error during evaluation: %s", e)
        raise


def main():
    try:
        x_test = pd.read_csv("data/preprocess/x_test_processed.csv")
        y_test = pd.read_csv("data/preprocess/y_test_processed.csv")

        model_path = "artifacts/xgb_model.pkl"

        results = evaluate_model(model_path, x_test, y_test)
        logger.debug(f"Evaluation Results: {results}")
        
    except Exception as e:
        logger.error("Unexpected error occurred !!!: %s",e)
        raise
    
if __name__=="__main__":
    main()
        


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
import yaml

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

def load_params(params_path:str)->dict:
    "Load params from params.yaml file"
    try:
        with open(params_path,"r") as f:
            params=yaml.safe_load(f)
        logger.debug("Parameters retrieved %s",params_path)
        return params
    except FileNotFoundError as e:
        logger.error("File not found %s",e)
        raise
    except yaml.YAMLError as e:
        logger.error("Yaml error: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred:%s",e)
        raise

def evaluate_model(model_path: str, x_test: pd.DataFrame, y_test: pd.DataFrame,x_train:pd.DataFrame,y_train:pd.DataFrame,params):
    """
    Evaluate trained model, log metrics and confusion matrix to MLflow.
    """
    try:
        logger.info("Starting model evaluation...")

        with mlflow.start_run(run_name="XGBoost_Evaluation"):
            
            mlflow.log_params(params["model_training"]["params"])

            # Load model
            with open(model_path,"rb") as f:
                model=pickle.load(f)
            logger.info("Model loaded successfully.")

            # Predictions
            y_pred_test = model.predict(x_test)
            y_pred_train=model.predict(x_train)
            

            # Metrics
            acc_test = accuracy_score(y_test, y_pred_test)
            acc_train=accuracy_score(y_train,y_pred_train)
            prec = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)


            metrics = {
                "accuracy_train": acc_train,
                "accuracy_test":acc_test,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,

            }
            
            os.makedirs("artifacts",exist_ok=True)
            with open("artifacts/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            # Log metrics to MLflow
            mlflow.log_metric("accuracy_train",acc_train)
            mlflow.log_metric("accuracy_test",acc_test)
            mlflow.log_metric("precision",prec)
            mlflow.log_metric("recall",rec)
            mlflow.log_metric("f1_score",f1)
          
            mlflow.log_artifact("artifacts/metrics.json")
            
            logger.debug("Metrics logged to MLflow.%s",metrics)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_test)
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

            logger.info(f"Evaluation done | Accuracy: {acc_test:.4f}, F1: {f1:.4f}")
            return metrics

    except Exception as e:
        logger.error(" Error during evaluation: %s", e)
        raise


def main():
    try:
        params=load_params(params_path="params.yaml")
        x_test = pd.read_csv("data/preprocess/x_test_processed.csv")
        y_test = pd.read_csv("data/preprocess/y_test_processed.csv")
        x_train=pd.read_csv("data/preprocess/x_train_processed.csv")
        y_train=pd.read_csv("data/preprocess/y_train_processed.csv")

        model_path = "artifacts/xgb_model.pkl"

        results = evaluate_model(model_path, x_test, y_test,x_train,y_train,params)
        logger.debug(f"Evaluation Results: {results}")
        
    except Exception as e:
        logger.error("Unexpected error occurred !!!: %s",e)
        raise
    
if __name__=="__main__":
    main()
        


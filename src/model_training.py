import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
import logging
import os
import pickle
import yaml

mlflow.set_tracking_uri("file:///C:/Users/Dell/OneDrive - Havells/Desktop/Classify-Spotify-Track-Genre/mlruns")
# local MLflow UI
# mlflow.set_tracking_uri("s3://your-bucket-name/mlflow")

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

#logging configration
logger=logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"model_training.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')

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


def load_data():
    try:
        x_train = pd.read_csv("data/preprocess/x_train_processed.csv")
        y_train = pd.read_csv("data/preprocess/y_train_processed.csv")
        logger.debug("Data loaded successfully")
        return x_train, y_train, 
    except Exception as e:
        logger.error(f" Error loading data: {e}")
        raise


def train_and_log_model(x_train, y_train,params):
    try:
 
        mlflow.set_experiment("XGBoost_Classification")
        mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)

    # Start MLflow run
        with mlflow.start_run():
            model_params=params["model_training"]["params"]
        # Model setup
            
            model = XGBClassifier(**model_params)
            model.fit(x_train, y_train)
        
        #log parameters
            mlflow.log_params(params)
            os.makedirs("artifacts", exist_ok=True)
            model_path = "artifacts/xgb_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.xgboost.log_model(model, "model",artifact_path="models",registered_model_name="xgboost_classifier")
        
            logger.debug("Training of model done successfully !!!")
            return model_path
    except Exception as e:
        logger.error("Unexpected error occur: %s",e)
        
        
def main():
    try:
        params=load_params(params_path="params.yaml")
        x_train,y_train=load_data()
        train_and_log_model(x_train,y_train,params)
        logger.debug("Training done successfully !!!")
    except Exception as e:
        logger.error("Unexpected error occur:%s",e)
        
        
if __name__=="__main__":
    main()
    



        


        

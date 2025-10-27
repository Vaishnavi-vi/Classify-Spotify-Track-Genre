import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
import logging
import os
import pickle


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


def load_data():
    try:
        x_train = pd.read_csv("data/preprocess/x_train_processed.csv")
        y_train = pd.read_csv("data/preprocess/y_train_processed.csv")
        logger.debug("Data loaded successfully")
        return x_train, y_train, 
    except Exception as e:
        logger.error(f" Error loading data: {e}")
        raise


def train_and_log_model(x_train, y_train):
    try:
 
        mlflow.set_experiment("XGBoost_Classification")

    # Start MLflow run
        with mlflow.start_run():
        # Model setup
            params = {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 3,
            "reg_alpha": 2,
            "gamma": 0.2,
            "random_state": 42}
        

            model = XGBClassifier(**params)
            model.fit(x_train, y_train)
        
        #log parameters
            mlflow.log_params(params)
            os.makedirs("artifacts", exist_ok=True)
            model_path = "artifacts/xgb_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.xgboost.log_model(model, "model")
        
            logger.debug("Training of model done successfully !!!")
            return model_path
    except Exception as e:
        logger.error("Unexpected error occur: %s",e)
        
        
def main():
    try:
        x_train,y_train=load_data()
        train_and_log_model(x_train,y_train)
        logger.debug("Training done successfully !!!")
    except Exception as e:
        logger.error("Unexpected error occur:%s",e)
        
        
if __name__=="__main__":
    main()


        


        

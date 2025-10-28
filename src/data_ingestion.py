import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import logging
import pymongo
import yaml


#Ensure the log_dir exists
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

#logging configration
logger=logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"data_ingestion.log")
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


def load_data(data_url:str)->pd.DataFrame:
    """Load data from csv file"""
    try:
        df=pd.read_csv(data_url,engine="python",on_bad_lines="skip")
        logger.debug("Data loaded from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file :%s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred %s",e) 
        raise
    
def pre_process(df:pd.DataFrame)->pd.DataFrame:
    """ Preprocess the Data"""
    try:
        df.drop(["track_id","artists","album_name","track_name","Unnamed: 0"],axis=1,inplace=True)
        logger.debug("Data Preprocessing Completed")
        return df
    except KeyError as e:
        logger.error("Missing Columns in the Data Frame %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred %s",e)
        raise
    
def upload_to_mongodb(df:pd.DataFrame,params:dict):
    try:
        CONNECTION_URL=params["mongodb"]["CONNECTION_URL"]
        DB_NAME=params["mongodb"]["DB_NAME"]
        COLLECTION_NAME=params["mongodb"]["COLLECTION_NAME"]
        
        client=pymongo.MongoClient(CONNECTION_URL)
        data_base=client[DB_NAME]
        
        if COLLECTION_NAME in data_base.list_collection_names():
            data_base.drop_collection(COLLECTION_NAME)
            logger.debug("Old collection data dropped scuucessfully!!!")
            
        collection=data_base[COLLECTION_NAME]
        logger.debug("Comnnected to mongodb atlas")
        records=df.to_dict(orient="records")
        collection.insert_many(records)
        logger.debug("All data uploaded to mongodb")
    except Exception as e:
        logger.error("Error uploading to mongodb")
        raise
    
def load_from_mongodb(params:dict)->pd.DataFrame:
    try:
        CONNECTION_URL=params["mongodb"]["CONNECTION_URL"]
        DB_NAME=params["mongodb"]["DB_NAME"]
        COLLECTION_NAME=params["mongodb"]["COLLECTION_NAME"]
        
        client=client=pymongo.MongoClient(CONNECTION_URL)
        data_base=client[DB_NAME]
        collection=data_base[COLLECTION_NAME]
        logger.debug("Comnnected to mongodb atlas for loading data")
        
        data=list(collection.find())
        df=pd.DataFrame(data)
        if "_id" in df.columns:
            df.drop("_id", axis=1, inplace=True)
        logger.debug("Data loaded from mongodb successfully!!!")
        return df
    except Exception as e:
        logger.error("Error loading data from nongodb :%s",e) 
        raise
    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test dataset"""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        
        # Create both 'data' and 'data/raw' directories if they don’t exist
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred %s", e)
        raise
    
def main():
    try:
        params=load_params(params_path="params.yaml")
        test_size=params["data_ingestion"]["test_size"]
        data_url="C:\\Users\\Dell\\OneDrive - Havells\\Desktop\\Classify-Spotify-Track-Genre\\Experiments\\dataset.csv"
        df=load_data(data_url)
        final_df=pre_process(df)
        upload_to_mongodb(final_df,params)
        df_from_mongodb=load_from_mongodb(params)
        train_data,test_data=train_test_split(df_from_mongodb,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path="./data")
        logger.debug("Data Ingestion Completed")
    except Exception as e:
        logger.error("Unexpected error occurred:%s",e)
        raise
    
if __name__=="__main__":
    main()
    
    
    

    
        

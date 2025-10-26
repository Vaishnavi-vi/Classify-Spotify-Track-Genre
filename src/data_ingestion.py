import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import logging
# import pymongo


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
        test_size=0.2
        data_url="C:\\Users\\Dell\\OneDrive - Havells\\Desktop\\Classify-Spotify-Track-Genre\\Experiments\\dataset.csv"
        df=load_data(data_url)
        final_df=pre_process(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path="./data")
        logger.debug("Data Ingestion Completed")
    except Exception as e:
        logger.error("Unexpected error occurred:%s",e)
        raise
    
if __name__=="__main__":
    main()
    

    
        

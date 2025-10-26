import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import pickle
import json
import seaborn as sns

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_selection.py")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "feature_selection.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(train_data_url:str,test_data_url:str)->pd.DataFrame:
    """ Load the raw data from data/raw"""
    try:
        train_data=pd.read_csv(train_data_url)
        test_data=pd.read_csv(test_data_url)
        logger.debug("Loaded data Properly")
        return train_data,test_data
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file :%s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred %s",e) 
        raise
    
def explicit(train_data:pd.DataFrame,test_data:pd.DataFrame,column_name="explicit")->pd.DataFrame:
    """ Map the explicit as 0 and 1"""
    try:
        mapp={False:0,True:1}
        train_data[column_name]=train_data[column_name].map(mapp)
        test_data[column_name]=test_data[column_name].map(mapp)
        logger.debug("Mapping for explicit done successfully !!!")
        return train_data,test_data
    except Exception as e:
        logger.error("Unexpected error occur:%s",e)
        raise
    
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

def apply_power_transformer(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Apply PowerTransformer to skewed numeric columns."""
    try:
        more_skewed = ["duration_ms", "loudness", "speechiness", "instrumentalness", "liveness", "time_signature"]

        preprocess = ColumnTransformer(
            transformers=[
                ("trf1", PowerTransformer(), more_skewed)
            ],
            remainder="passthrough"
        )

        # Fit on train and transform both datasets
        train_scaled = preprocess.fit_transform(train_data)
        test_scaled = preprocess.transform(test_data)

        # Rebuild DataFrame with correct column order
        
        train_scaled_df = pd.DataFrame(train_scaled, columns=train_data.columns, index=train_data.index)
        test_scaled_df = pd.DataFrame(test_scaled, columns=test_data.columns, index=test_data.index)

        logger.debug("PowerTransformer applied successfully on skewed columns!")
        return train_scaled_df, test_scaled_df

    except Exception as e:
        logger.error("Error applying PowerTransformer: %s", e)
        raise

    
def feature_selection(train_scaled_df:pd.DataFrame,test_scaled_df:pd.DataFrame,save_dir="artifacts",file_path="mi_feature_importance.png"):
    """ Select the best k feature"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        x_train=train_scaled_df.iloc[:,:-1]
        y_train=train_scaled_df["track_genre"]
        mi_scores=mutual_info_classif(x_train,y_train,discrete_features="auto")
        mi_df=pd.DataFrame({"Features":x_train.columns,"MI_Score":mi_scores})
        mi_df_df=mi_df.sort_values(by="MI_Score",ascending=False)
        
        plt.figure(figsize=(20, 6))
        sns.barplot(x="Features", y="MI_Score", data=mi_df_df, palette="viridis")
        plt.title("Top 15 Features by Mutual Information")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mi_feature_importance.png"))
        plt.close()

        logger.debug("Feature selection and visualization done successfully!")
        return mi_df_df

    except Exception as e:
        logger.error("Unexpected error in feature_selection: %s", e)
        raise
    

def drop_columns(train_scaled_df:pd.DataFrame,test_scaled_df:pd.DataFrame)->pd.DataFrame:
    """  Drop Columns WHde m_classif valueis less """
    try:
        train_scaled_df.drop(["key","explicit"],axis=1,inplace=True)
        test_scaled_df.drop(["key","explicit"],axis=1,inplace=True)
        logger.debug("Dropping less important for model ")
        return train_scaled_df,test_scaled_df
    except Exception as e:
        logger.error("Unexpected error in feature_selection: %s", e)
        raise
    
def scaling(train_scaled_df: pd.DataFrame, test_scaled_df: pd.DataFrame,save_dir="artifacts"):
    """Apply StandardScaler and SMOTE, then return DataFrames (not numpy arrays)."""
    try:
        os.makedirs(save_dir,exist_ok=True)
        scaler = StandardScaler()
        smote = SMOTE(random_state=42)
        
        with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        x_train = train_scaled_df.drop(columns=["track_genre"])
        y_train = train_scaled_df["track_genre"]
        x_test = test_scaled_df.drop(columns=["track_genre"])
        y_test = test_scaled_df["track_genre"]

        # Scale data
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Convert back to DataFrames
        x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
        x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

        # Apply SMOTE
        x_train_resample, y_train_resample = smote.fit_resample(x_train_scaled_df, y_train)

        # Convert SMOTE outputs back to DataFrames (SMOTE returns numpy arrays)
        x_train_resample_df = pd.DataFrame(x_train_resample, columns=x_train.columns)
        y_train_resample_df = pd.DataFrame(y_train_resample, columns=["track_genre"])

        logger.debug("Scaling and SMOTE applied successfully (DataFrames retained).")

        return x_train_resample_df, y_train_resample_df, x_test_scaled_df, pd.DataFrame(y_test, columns=["track_genre"])
    except Exception as e:
        logger.error("Error in scaling or SMOTE: %s", e)
        raise

        

def save_data(df: pd.DataFrame, file_name: str, save_dir="data/preprocess"):
    """Save a DataFrame to CSV inside artifacts folder."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        df.to_csv(file_path, index=False)
        logger.debug(f"Data saved successfully at {file_path}")
    except Exception as e:
        logger.error("Error saving data: %s", e)
        raise
          
def main():
    """  """
    try:
        train_data,test_data=load_data(train_data_url="data/feature_engineering/train.csv",test_data_url="data/feature_engineering/test.csv")
        train_data,test_data=explicit(train_data,test_data)
        train_scaled_df, test_scaled_df=apply_power_transformer(train_data,test_data)
        feature_selection(train_scaled_df,test_scaled_df)
        train_scaled_df,test_scaled_df=drop_columns(train_scaled_df,test_scaled_df)
        x_train_resample_df, y_train_resample_df, x_test_scaled_df, y_test=scaling(train_scaled_df,test_scaled_df)
        save_data(x_train_resample_df,"x_train_processed.csv")
        save_data(y_train_resample_df,"y_train_processed.csv")
        save_data(x_test_scaled_df,"x_test_processed.csv")
        save_data(y_test,"y_test_processed.csv")
        logger.debug("Feature Selection and preprocessing done successfully !")
    except Exception as e:
        logger.error("Unexpected error in feature_selection: %s", e)
        raise
        
    
    
if __name__=="__main__":
    main()
        

       
        

    
    

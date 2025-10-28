import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import pickle
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

# -------------------- Logging Setup --------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_selection.py")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "feature_selection.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(train_data_url: str, test_data_url: str):
    """ Load the raw data from data/raw"""
    try:
        train_data = pd.read_csv(train_data_url)
        test_data = pd.read_csv(test_data_url)
        logger.debug("Loaded data successfully.")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def explicit(train_data, test_data, column_name="explicit"):
    """ Map the explicit column to 0/1 """
    try:
        mapp = {False: 0, True: 1}
        train_data[column_name] = train_data[column_name].map(mapp)
        test_data[column_name] = test_data[column_name].map(mapp)
        logger.debug("Mapping for explicit done successfully.")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error in explicit mapping: {e}")
        raise

def power_transformer(train_data,test_data):
    """ Apply power transformer on more_skewed columns"""
    try:
        x_train=train_data.drop(["track_genre"],axis=1)
        x_test=test_data.drop(["track_genre"],axis=1)
        y_train=train_data["track_genre"]
        y_test=test_data["track_genre"]
        
        skewed_col=["duration_ms","loudness","speechiness","instrumentalness","liveness","time_signature"]
        x_train[skewed_col] = x_train[skewed_col].astype(float)
        x_test[skewed_col] = x_test[skewed_col].astype(float)

        pt=PowerTransformer()
        pt.fit(x_train[skewed_col])
        
        x_train.loc[:,skewed_col]=pt.transform(x_train[skewed_col])
        x_test.loc[:,skewed_col]=pt.transform(x_test[skewed_col])
        
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/power_transformer.pkl", "wb") as f:
            pickle.dump(pt, f)
            
        logger.debug("PowerTransformer applied successfully.")
        return x_train,y_train,x_test,y_test
    except Exception as e:
        logger.error(f"Error applying PowerTransformer: {e}")
        raise

def feature_selection(x_train,y_train, save_dir="artifacts"):
    """Select the best k features using mutual info."""
    try:
        
        os.makedirs(save_dir, exist_ok=True)
        mi_scores = mutual_info_classif(x_train, y_train,random_state=42)
        mi_df = pd.DataFrame({"Features": x_train.columns, "MI_Score": mi_scores})
        mi_df = mi_df.sort_values(by="MI_Score", ascending=False)

        plt.figure(figsize=(20, 6))
        sns.barplot(x="Features", y="MI_Score", data=mi_df, palette="viridis")
        plt.title("Features by Mutual Information")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mi_feature_importance.png"))
        plt.close()

        logger.debug("Feature selection and visualization done successfully.")
        return mi_df
    except Exception as e:
        logger.error(f"Error in feature_selection: {e}")
        raise


def drop_columns(x_train, x_test):
    """Drop less important columns."""
    try:
        x_train.drop(["key","explicit"],axis=1, inplace=True)
        x_test.drop(["key","explicit"],axis=1,inplace=True)
        logger.debug("Dropped less important columns successfully.")
        return x_train,x_test
    except Exception as e:
        logger.error(f"Error in drop_columns: {e}")
        raise


def scaling(x_train,x_test,y_train,y_test, save_dir="artifacts"):
    """Apply StandardScaler and SMOTE."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        scaler = StandardScaler()
        smote = SMOTE(random_state=42)

        x_train_scaled = scaler.fit_transform(x_train)       
        x_test_scaled = scaler.transform(x_test)

        x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
        x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)

        x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled_df, y_train)

        with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        logger.debug("Scaling and SMOTE applied successfully.")
        return x_train_resampled, y_train_resampled, x_test_scaled_df, y_test
    except Exception as e:
        logger.error(f"Error in scaling: {e}")
        raise


def save_data(df, file_name, save_dir="data/preprocess"):
    """Save DataFrame to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, file_name), index=False)
    logger.debug(f"Saved {file_name} successfully.")


def main():
    try:
        train_data, test_data = load_data("data/feature_engineering/train.csv", "data/feature_engineering/test.csv")
        train_data, test_data = explicit(train_data, test_data)
        x_train,y_train,x_test,y_test=power_transformer(train_data,test_data)
        feature_selection(x_train,y_train)
        x_train,x_test=drop_columns(x_train,x_test)
        x_train_resampled, y_train_resampled, x_test_scaled_df, y_test = scaling(x_train,x_test,y_train,y_test)

        save_data(x_train_resampled, "x_train_processed.csv")
        save_data(pd.DataFrame(y_train_resampled, columns=["track_genre"]), "y_train_processed.csv")
        save_data(x_test_scaled_df, "x_test_processed.csv")
        save_data(pd.DataFrame(y_test, columns=["track_genre"]), "y_test_processed.csv")

        logger.debug("Feature Selection and preprocessing done successfully!")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise


if __name__ == "__main__":
    main()

        

       
        

    
    

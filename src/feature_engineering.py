import gensim
import gensim.downloader as api
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
import json
import seaborn as sns

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(train_data_url:str,test_data_path:str)->pd.DataFrame:
    """ Load the raw data from data/raw"""
    try:
        train_data=pd.read_csv(train_data_url)
        test_data=pd.read_csv(test_data_path)
        logger.debug("Loaded data Properly")
        return train_data,test_data
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file :%s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred %s",e) 
        raise

def normalise_genre(train_data,test_data,column="track_genre"):
    try:
        mapping = {
        "black-metal": "metal",
        "death-metal": "metal",
        "j-rock": "rock",
        "psych-rock": "rock",
        "deep-house": "house",
        "chicago-house": "house",
        "progressive-house": "house",
        "minimal-techno": "techno",
        "detroit-techno": "techno",
        "drum-and-bass": "drum",
        "pop-film": "pop",
        "j-dance": "dance",
        "j-idol": "idol",
        "r-n-b": "rnb",
        "show-tunes": "tunes",
        "world-music": "music"
    }
        train_data[column] = train_data[column].replace(mapping)
        test_data[column]=test_data[column].replace(mapping)
        logger.debug("Normalize the Track_genre Successfully!!!: %s",mapping)
        return train_data,test_data
    except Exception as e:
        logger.error("Unexpected error occurred %s",e) 
        raise
        

def generate_embeddings(train_df, test_df,column="track_genre"):
    try:
        glove_model=api.load("glove-wiki-gigaword-300")
        new_genre_train=train_df[column].unique()
        new_genre_test=test_df[column].unique()
        vector_train=[glove_model[word] for word in new_genre_train]
        vector_test=[glove_model[word] for word in new_genre_test]
        vector_df_train=pd.DataFrame(dict(zip(new_genre_train,vector_train)))
        vector_df_train=vector_df_train.T
        vector_df_test=pd.DataFrame(dict(zip(new_genre_test,vector_test)))
        vector_df_test=vector_df_test.T
        logger.debug("Generated the embedding successfully !!!")
        return vector_df_train,vector_df_test
    except Exception as e:
        logger.error("Unexpected error occurred!!",e)
    
    
def scale_pca(train_vec:pd.DataFrame,test_vec:pd.DataFrame)->pd.DataFrame:
    """ Applying Standard Scalar and PCA """
    try:
        scalar=StandardScaler()
        df_train=scalar.fit_transform(train_vec)
        df_test=scalar.fit_transform(test_vec)
        df_train_norm=pd.DataFrame(df_train,columns=list(map(str,range(1,301))),index=train_vec.index)
        df_test_norm=pd.DataFrame(df_test,columns=list(map(str,range(1,301))),index=test_vec.index)
        pca=PCA(n_components=2)
        pca_df_train=pca.fit_transform(df_train_norm)
        pca_df_test=pca.fit_transform(df_test_norm)
        pca_df_new_train=pd.DataFrame(pca_df_train,columns=["PCA1","PCA2"],index=train_vec.index)
        pca_df_new_test=pd.DataFrame(pca_df_test,columns=["PCA1","PCA2"],index=test_vec.index)
        logger.debug("Standard Scalar and PCA done Successfully!!!")
        return pca_df_new_train,pca_df_new_test
    except Exception as e:
        logger.error("Unexpected Error Occur: %s",e)
        raise

def find_optimal_k(pca_df_train,save_dir="artifacts",file_path="Optimal_k.png"):
    """ Visualize Yellowbrick to find the optimal number of Kmeans cluster """
    try:
        os.makedirs(save_dir,exist_ok=True)
        save_path=os.path.join(save_dir,file_path)
        visualizer=KElbowVisualizer(KMeans(random_state=42),k=(2,11),timings=False,metric="distortion")
        visualizer.fit(pca_df_train)
        visualizer.show(outpath=save_path, clear_figure=True)
        plt.close() 
        
        optimal_k=visualizer.elbow_value_
        logger.debug("Visualization of plot done successsfully!!!")
        return optimal_k
    except Exception as e:
        logger.error("Unexpected rror occurred: %s",e)
        raise


def cluster_metrics(optimal_k, pca_df_train,save_dir="artifacts"):
    """
    Perform clustering with KMeans, calculate silhouette scores (Manhattan & Euclidean),
    save the metrics to artifacts/cluster_metrics, and visualize clusters.
    """
    try:
        os.makedirs(save_dir,exist_ok=True)
        save_metrics = os.path.join(save_dir, "cluster_metrics.json")
        save_plots = os.path.join(save_dir, "cluster_plots.png")

        mean = KMeans(n_clusters=optimal_k, random_state=42)
        mean.fit(pca_df_train)

        
        manhattan_score = silhouette_score(pca_df_train, mean.labels_, metric="manhattan")
        euclidean_score = silhouette_score(pca_df_train, mean.labels_, metric="euclidean")

       
        metrics = {
            "optimal_k": int(optimal_k),
            "manhattan_score": float(manhattan_score),
            "euclidean_score": float(euclidean_score)
        }

        with open(save_metrics, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.debug("Silhouette scores saved to:%s",save_metrics)

       
        pca_df_train["cluster"] = mean.labels_
           
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=pca_df_train["PCA1"], y=pca_df_train["PCA2"], hue=pca_df_train["cluster"], palette="viridis", s=50)
        plt.scatter(mean.cluster_centers_[:, 0], mean.cluster_centers_[:, 1],
                    marker="x", color="red", s=120, label="Centroid")
        plt.legend()
        plt.title("KMeans Clustering on PCA Components")
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")

       
        plt.savefig(save_plots, bbox_inches="tight")
        plt.close()
        return f"manhattan_score:{manhattan_score}, euclidean_score:{euclidean_score}"

    except Exception as e:
        logger.error(" Unexpected error in feature_engineering3: %s", e)
        raise
    
            
def mapping(train_data,test_data,column="track_genre"):
    try:
        genre_to_cluster = {
        # Cluster 0
        'acoustic': 0, 'ambient': 0, 'anime': 0, 'metal': 0, 'bluegrass': 0, 'blues': 0, 'chill': 0, 
        'classical': 0, 'comedy': 0, 'dance': 0, 'techno': 0, 'disco': 0, 'drum': 0, 'dub': 0, 
        'electro': 0, 'electronic': 0, 'folk': 0, 'funk': 0, 'garage': 0, 'gospel': 0, 'groove': 0, 
        'guitar': 0, 'hardcore': 0, 'hip-hop': 0, 'indie': 0, 'idol': 0, 'rock': 0, 'jazz': 0, 
        'piano': 0, 'pop': 0, 'punk': 0, 'reggae': 0, 'rockabilly': 0, 'salsa': 0, 'samba': 0, 
        'tunes': 0, 'singer-songwriter': 0, 'ska': 0, 'songwriter': 0, 'soul': 0, 'tango': 0, 
        'trance': 0, 'music': 0,

        # Cluster 1
        'afrobeat': 1, 'alt-rock': 1, 'breakbeat': 1, 'cantopop': 1, 'dancehall': 1, 'dubstep': 1, 
        'edm': 1, 'emo': 1, 'forro': 1, 'goth': 1, 'grindcore': 1, 'grunge': 1, 'hard-rock': 1, 
        'hardstyle': 1, 'heavy-metal': 1, 'honky-tonk': 1, 'idm': 1, 'indie-pop': 1, 'j-pop': 1, 
        'k-pop': 1, 'mandopop': 1, 'metalcore': 1, 'mpb': 1, 'new-age': 1, 'pagode': 1, 'power-pop': 1, 
        'punk-rock': 1, 'rnb': 1, 'reggaeton': 1, 'rock-n-roll': 1, 'sertanejo': 1, 'synth-pop': 1, 
        'trip-hop': 1,

        # Cluster 2
        'alternative': 2, 'brazil': 2, 'british': 2, 'house': 2, 'children': 2, 'club': 2, 'country': 2, 
        'disney': 2, 'french': 2, 'german': 2, 'happy': 2, 'indian': 2, 'industrial': 2, 'iranian': 2, 
        'kids': 2, 'latin': 2, 'latino': 2, 'malay': 2, 'opera': 2, 'party': 2, 'romance': 2, 'sad': 2, 
        'sleep': 2, 'spanish': 2, 'study': 2, 'swedish': 2, 'turkish': 2}
        
        train_data[column]=train_data[column].map(genre_to_cluster)
        test_data[column]=test_data[column].map(genre_to_cluster)
        logger.debug("Mapping Done Successfully !")
        return train_data,test_data
    except Exception as e:
        logger.error("Unexpected error occurred:%s",e)
        raise
    

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path= "data") -> None:
    """Save train and test dataset"""
    try:
        featureing_engineering_data_path = os.path.join(data_path, "feature_engineering")
        
        # Create both 'data' and 'data/raw' directories if they don’t exist
        os.makedirs(featureing_engineering_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(featureing_engineering_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(featureing_engineering_data_path, "test.csv"), index=False)
        
        logger.debug("Train and test data saved to %s", featureing_engineering_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred %s", e)
        raise
                

def main():
    try:
        train_data,test_data=load_data(train_data_url="data/raw/train.csv",test_data_path="data/raw/test.csv")
        train_data,test_data=normalise_genre(train_data,test_data)
        vector_df_train,vector_df_test=generate_embeddings(train_data,test_data)
        pca_df_new_train,pca_df_new_test=scale_pca(vector_df_train,vector_df_test)
        optimal_k=find_optimal_k(pca_df_new_train)
        cluster_metrics(optimal_k,pca_df_new_train)
        train_data,test_data=mapping(train_data,test_data)
        save_data(train_data,test_data)
        
        logger.debug("All Process done Successfully!!!")
    except Exception as e:
        logger.error("Unexpected error occurred: %s",e)
        raise
    

if __name__=="__main__":
    main()
        
        


        
        




        
        

        
        

        
        
  
    

    
        

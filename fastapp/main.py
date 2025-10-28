from fastapi import FastAPI
import pickle
from fastapi.responses import JSONResponse
import pandas as pd
import logging
import os
from fastapp.input.user_input import user_input
from fastapp.input.predict import prediction


app = FastAPI(
    title="Track Genre Classifier",
    description="Predicts the genre cluster of a track based on its audio features.",
    version="1.0"
)

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

#logging configration
logger=logging.getLogger("genre_ai")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"genre_api.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
  
@app.get("/about")
def view():
    return {"message":"Track Genre Classification API is running!"}


@app.post("/predict")
def predict(value:user_input):
    try:
    
        input_data={"popularity":value.popularity,
                              "duration_ms":value.duration_ms,
                              "danceability":value.danceability,
                              "explicit":value.explicit,
                              "energy":value.energy,
                              "key":value.key,
                              "loudness":value.loudness,
                              "mode":value.mode,
                              "speechiness":value.speechiness,
                              "acousticness":value.acousticness,
                              "instrumentalness":value.instrumentalness,
                              "liveness":value.liveness,
                              "valence":value.valence,
                              "tempo":value.tempo,
                              "time_signature":value.time_signature}
    
        final_prediction=prediction(input_data)
        logger.info("Prediction Successfull:")

        return JSONResponse(status_code=200,content={"result":final_prediction})
    except Exception as e:
        logger.error("Error occur:%s",e)
        return JSONResponse(status_code=500,content={"message":str(e)})
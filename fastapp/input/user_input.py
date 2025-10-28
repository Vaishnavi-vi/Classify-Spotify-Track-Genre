from pydantic import BaseModel,Field
from typing import Annotated



class user_input(BaseModel):
    popularity:Annotated[float,Field(...,description="Enter the Popularity of the song",gt=0.0,lt=100.0)]
    duration_ms:Annotated[float,Field(...,description="Enter the duration of the song")]
    explicit:Annotated[float,Field(...,desciption="Enter the explicit")]
    danceability:Annotated[float,Field(...,description="Enter the danceability of the song",gt=0,lt=1.0)]
    energy:Annotated[float,Field(...,description="Enter the energy scale of the song",gt=0.0,lt=1.0)]
    key:Annotated[float,Field(...,description="Enter the key")]
    loudness:Annotated[float,Field(...,description='Enter the loudness of the song',gt=-20.0,lt=20.0)]
    mode:Annotated[float,Field(...,description="Enter the mode of the song either 0 or 1")]
    speechiness:Annotated[float,Field(...,description="Enter the spechiness of the song",gt=0.00,lt=0.15)]
    acousticness:Annotated[float,Field(...,description="Enter the accousticness of the song",gt=0.00,lt=1.00)]
    instrumentalness:Annotated[float,Field(...,description="Enter the instrumentalness of the song",gt=0.00,lt=0.12)]
    liveness:Annotated[float,Field(...,description="Enter the liveness of the song",gt=0.00,lt=0.53)]
    valence:Annotated[float,Field(...,description="Enter the valence",gt=0.0,lt=1.0)]
    tempo:Annotated[float,Field(...,description="Enter the tempo of the song",gt=0,lt=250)]
    time_signature:Annotated[float,Field(...,description="Enter the time_signature value",gt=-1,lt=6)]
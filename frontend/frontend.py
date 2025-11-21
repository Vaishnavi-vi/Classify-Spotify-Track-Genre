import streamlit as st
import pickle
from PIL import Image
import requests

# FastAPI endpoint
url_link="http://fastapi:8000/predict"
# url_link="http://127.0.0.1:8000/predict" local run 


# ---------------------------
# SIDEBAR DESIGN
# ---------------------------
st.sidebar.title("🎧 Track Genre Classifier")

# Logo / App Image
try:
    logo = Image.open("C:\\Users\\Dell\\Downloads\\image.png")
    st.sidebar.image(logo, use_container_width=True)
except:
    st.sidebar.write("")

# API Health Check
st.sidebar.subheader("API Health Status")
try:
    health = requests.get("http://127.0.0.1:8000/health")
    if health.status_code == 200:
        st.sidebar.success("API Running")
    else:
        st.sidebar.error("API Not Responding")
except:
    st.sidebar.error("API Offline")

# Navigation
page = st.sidebar.radio("Navigation", ["Home", "Audio Genre Prediction"])

# About Section
st.sidebar.markdown("---")
st.sidebar.subheader("About App")
st.sidebar.info(
    "This app predicts **Spotify Track Genres** using machine learning.\n\n"
    "Built with **FastAPI + Streamlit** "
)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("💡 *Created by Vaishnavi Barolia*")


if page == "Home":
    st.header("🎧 Classify Track Genre — Spotify App")
    image = Image.open("C:\\Users\\Dell\\Downloads\\image.png")
    st.image(image, use_container_width=True)
    st.write("""
    Welcome to the **Track Genre Predictor**!  
    Use the sidebar to navigate to the prediction page and enter track features.
    """)

elif page == "Audio Genre Prediction":
    st.header("🎵 Real-Time Track Genre Prediction")
    st.subheader("Tune the values below to predict genre:")

    # Input fields
    popularity = st.number_input("Popularity", min_value=0.0, max_value=99.0, step=5.0)
    duration_ms = st.number_input("Duration (ms)", min_value=8560.0, max_value=5237295.0, step=0.1)
    explicit = st.number_input("Explicit (0 or 1)", min_value=0, max_value=1)
    danceability = st.number_input("Danceability", min_value=0.0, max_value=1.0, step=0.02)
    energy = st.number_input("Energy", min_value=0.0, max_value=1.0, step=0.10)
    key = st.number_input("Key", min_value=0, max_value=11)
    loudness = st.number_input("Loudness", min_value=-60, max_value=5, step=1)
    mode = st.number_input("Mode", min_value=0, max_value=1)
    speechiness = st.number_input("Speechiness", min_value=0.00, max_value=0.15, step=0.03)
    acousticness = st.number_input("Acousticness", min_value=0.00, max_value=1.00)
    instrumentalness = st.number_input("Instrumentalness", min_value=0.00, max_value=0.11)
    liveness = st.number_input("Liveness", min_value=0.00, max_value=0.53)
    valence = st.number_input("Valence", min_value=0.00, max_value=1.0)
    tempo = st.number_input("Tempo", min_value=0, max_value=250, step=10)
    time_signature = st.number_input("Time Signature", min_value=0, max_value=6)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Predicting genre…"):
            input_data = {
                "popularity": popularity,
                "duration_ms": duration_ms,
                "explicit": explicit,
                "danceability": danceability,
                "energy": energy,
                "key": key,
                "loudness": loudness,
                "mode": mode,
                "speechiness": speechiness,
                "acousticness": acousticness,
                "instrumentalness": acousticness,
                "liveness": liveness,
                "valence": valence,
                "tempo": tempo,
                "time_signature": time_signature,
            }

            try:
                response = requests.post(url_link, json=input_data)
                if response.status_code in [200, 201, 202]:
                    output = response.json()
                    st.success("Prediction:")
                    st.write(f"🎼 **Predicted Genre:** `{output['result']}`")
                else:
                    st.warning(f"{response.status_code} → {response.text}")
            except Exception as e:
                st.error(e)

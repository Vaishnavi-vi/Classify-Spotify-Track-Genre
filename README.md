# 🎵 Spotify Track Genre Classification

This project is an **end-to-end machine learning pipeline** that classifies Spotify tracks into genres based on audio features.  
It uses:

- **Scikit-learn / AutoML** for model training  
- **FastAPI** for backend prediction API  
- **Streamlit** for interactive frontend  
- **Docker** for containerized deployment  
- **MLOps Pipeline** for preprocessing, training, evaluation, and deployment, ensuring a **reproducible and production-ready workflow**    

The system predicts the **genre** of a track using features like:

- Danceability  
- Energy  
- Loudness  
- Tempo  
- Instrumentalness  
- Acousticness  
- Speechiness  

---

## 📌 Project Architecture

```
User → Streamlit UI → FastAPI API → ML Model → Genre Prediction
```

### Components:
- **frontend/** → Streamlit app  
- **fastapp/** → FastAPI backend + model  
- **model/** → Saved ML model + scaler/encoder  
- **docker-compose.yml** → Runs both services together  

---

## 🚀 Features

- Predicts genre of a track using audio features  
- FastAPI `/predict` endpoint returns predicted genre  
- Streamlit UI for manual input or CSV upload  
- Dockerized for easy deployment  

---
## 🔧 How to Run

### 1️⃣ Clone Repository
```bash
git clone <repo-url>
cd Classify-Spotify-Track-Genre
```

### 2️⃣ Run with Docker
```bash
docker-compose up --build
```
- Streamlit UI: http://localhost:8501  
- FastAPI Docs: http://localhost:8000/docs  

### 3️⃣ Run Locally
#### FastAPI:
```bash
cd fastapp
uvicorn fastapp.main:app --reload 
```
#### Streamlit:
```bash
cd frontend
streamlit run frontend/frontend.py
```

---

## 🧠 Model Details

- **Model:** Random Forest / XGBoost / AutoML-selected  
- **Features:** Numeric audio features from Spotify dataset  
- **Preprocessing:** Scaling, encoding categorical variables  
- **Output:** Genre label  

---

## 📡 API Example

### Request:
```json
{
  "danceability": 0.8,
  "energy": 0.7,
  "tempo": 120,
  "acousticness": 0.1,
  "instrumentalness": 0.0
}
```

### Response:
```json
{
  "predicted_genre": "Pop"
}
```

---

## 🎨 Streamlit UI

- Input features manually or via CSV  
- Shows predicted genre  
- Simple interactive layout  

---

## 🛠 Future Enhancements
## 🚀 Features

- Predicts genre of a track using audio features  
- FastAPI `/predict` endpoint returns predicted genre  
- Streamlit UI for manual input or CSV upload  
- Dockerized for easy deployment  
- **MLOps pipeline** for preprocessing, training, evaluation, and deployment


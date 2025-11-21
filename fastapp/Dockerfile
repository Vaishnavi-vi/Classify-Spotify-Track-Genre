
# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app code
COPY fastapp/ ./fastapp/

# Copy only specific artifact files
COPY artifacts/scaler.pkl ./artifacts/scaler.pkl
COPY artifacts/xgb_model.pkl ./artifacts/xgb_model.pkl
COPY artifacts/power_transformer.pkl ./artifacts/power_transformer.pkl

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "fastapp.main:app", "--host", "0.0.0.0", "--port", "8000"]




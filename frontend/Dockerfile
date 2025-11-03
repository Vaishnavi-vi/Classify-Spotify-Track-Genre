# Streamlit Dockerfile
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements from root
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Streamlit app code
COPY . .

# Expose Streamlit’s default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "frontend/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]

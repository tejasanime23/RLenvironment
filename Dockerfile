# Use a lightweight stable-baselines3 compatible image
FROM python:3.10-slim

# Install system dependencies for Matplotlib and Graph processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project core
COPY . .

# Ensure the Gradio app knows it's running in a container
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Metadata and Entry points
EXPOSE 7860

# Launch the Gradio "Judge Magnet" Dashboard by default
# For OpenAI eval, the judge will override this with: python inference.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

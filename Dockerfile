# Use a lightweight Python image
FROM python:3.12-slim

# System dependencies for audio processing (Librosa/SoundFile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR $HOME/app

# 1. Install dependencies first (for better caching)
COPY --chown=user requirements.txt .
# Install CPU-specific torch to save massive amounts of time/space
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Copy the rest of the app
COPY --chown=user . .

# Expose the HF default port
EXPOSE 7860

# Run migrations, collect static files, and start Gunicorn
CMD python manage.py migrate && \
    python manage.py collectstatic --no-input && \
    gunicorn config.wsgi:application --bind 0.0.0.0:7860 --timeout 120
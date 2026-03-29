# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create a non-root user for security (HF requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Django project
COPY --chown=user . .

# Expose port 7860 (Hugging Face's default port)
EXPOSE 7860

# Run migrations and start the server using Gunicorn
# Replace 'myproject' with the folder name containing your wsgi.py
CMD python manage.py migrate && \
    gunicorn config.wsgi:application --bind 0.0.0.0:7860
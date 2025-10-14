# Use the Python 3.11 base image
FROM python:3.11-slim

# Set the Working Directory
WORKDIR /app

# Copy requirements.txt and install libraries
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining code (app.py, index.html, .pkl files)
COPY . .

# Download the NLTK data (during Docker Build)
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"
ENV NLTK_DATA /usr/local/share/nltk_data

# Run Gunicorn as an HTTP Server.
# Gunicorn will look for the Flask App named 'app' inside app.py.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
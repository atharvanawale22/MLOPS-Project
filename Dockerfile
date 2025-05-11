# Use Python base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy everything to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

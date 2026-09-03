# Use official Python 3.10 slim image (matches Hugging Face spaces env)
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for some ML packages (e.g., RDKit, CV2)
 RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies
# --no-cache-dir keeps the Docker image size small
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire web app directory into the container
COPY . .

# Expose the Gradio default port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
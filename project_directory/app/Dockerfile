# Use Python slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch and torchvision from PyTorch's stable repository
RUN pip install torch==2.0.0+cpu torchvision==0.15.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

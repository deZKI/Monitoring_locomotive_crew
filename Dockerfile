# Use the Python 3.11 Alpine base image
FROM python:3.11-alpine

# Set an environment variable to prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update the package repositories and install required system dependencies
RUN apk update && \
    apk add --no-cache build-base libffi-dev python3-dev jpeg-dev zlib-dev && \
    apk add --no-cache py3-cffi

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for the Django application
EXPOSE 8000:8000

# Define the command to run when the container starts
CMD ["python", "monitoring_crew/manage.py", "runserver", "0.0.0.0:8000"]

# Specify the base image with Python 3.10.12
FROM python:3.10.12-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements-lock.txt file to the /app/src directory
COPY requirements-lock.txt /app
COPY setup.py /app

# Copy your source code to a temporary location first
COPY ./src /app/src

# Install the Python package as an editable package
RUN pip install --no-cache-dir -e /app
# Specify the base image with Python 3.10.12
FROM python:3.10.12-slim-buster

# Automatically trims unmanaged memory in Dask workers at the cost of performance 
# due to continuous system calls.
# Copy the Dask configuration file into the container
COPY distributed.yaml /etc/dask/distributed.yaml

# Set the working directory in the container
WORKDIR /app

# Copy the requirements-lock.txt file to the /app/src directory
#COPY requirements-lock.txt /app
#COPY setup.py /app

# Copy your source code to a temporary location first
#COPY ./src /app/src

# Install the Python package as an editable package
#RUN pip install --no-cache-dir -e /app

RUN pip install robustraster==0.1.3
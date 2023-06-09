# FROM python:3.8-slim
FROM --platform=linux/amd64 python:3.8-slim

RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -f -p /usr/local/

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN pip install --upgrade pip 

RUN conda config --add channels conda-forge

RUN conda install humap

COPY . /app 
WORKDIR /app

RUN python minimal_test.py
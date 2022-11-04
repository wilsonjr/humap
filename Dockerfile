FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN pip install --upgrade pip 

RUN conda install numpy
RUN conda install scipy
RUN conda install scikit-learn
RUN conda install eigen
RUN conda install pybind11

COPY . /app 
WORKDIR /app

RUN python setup.py build_ext -I/opt/conda/include/eigen3 install 
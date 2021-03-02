FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
MAINTAINER Tabish Rashid

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN apt-get update
#RUN apt-get -y install python3
RUN apt update -y
#RUN apt install python3
#RUN cd /tmp && wget https://www.python.org/ftp/python/3.7.2/Python-3.7.2.tar.xz && tar -xf Python-3.7.2.tar.xz && cd Python-3.7.2 && ./configure --enable-optimizations
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
#RUN update-alternatives --config python3 < "2"
RUN apt install software-properties-common
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN yes | apt install python3.7
RUN rm /usr/bin/python3 | ln -s /usr/bin/python3.7 /usr/bin/python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Python packages we use (or used at one point...)
RUN python3 -m pip install numpy scipy pyyaml matplotlib
RUN python3 -m pip install imageio
RUN python3 -m pip install tensorboard-logger
RUN python3 -m pip install pygame

RUN mkdir /install
WORKDIR /install

RUN python3 -m pip install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN python3 -m pip install setuptools
RUN pip3 install --upgrade --force-reinstall setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN python3 -m pip install torch
RUN python3 -m pip install torchvision snakeviz pytest probscale

## -- SMAC
RUN python3 -m pip install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /pymarl/3rdparty/StarCraftII

WORKDIR /pymarl

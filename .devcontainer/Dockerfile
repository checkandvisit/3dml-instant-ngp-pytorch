# syntax=docker/dockerfile:1.3

# Other version not tested yet
ARG CUDA_VERSION=11.3.0
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Fix Cuda Path
ENV CUDA_ROOT /usr/local/cuda
ENV PATH $PATH:$CUDA_ROOT/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda/lib$LIBRARY_PATH

ENV DEPENDENCIES="/dependencies"
WORKDIR ${DEPENDENCIES}

RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

# basic packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update                && \
    apt-get -y upgrade               && \
    apt-get install -y                  \
    software-properties-common      \
    sudo                            \
    cmake                           \
    build-essential                 \
    wget                            \
    curl                            \
    git                             \
    swig

#-------------------------------------------------------------
#       Python Installation
#-------------------------------------------------------------

ARG PYTHON_VERSION=3.8

RUN add-apt-repository ppa:deadsnakes/ppa       && \
    apt-get -y update                           && \
    apt-get install -y                             \
    python${PYTHON_VERSION}                        \
    python${PYTHON_VERSION}-dev                    \
    libpython${PYTHON_VERSION}-dev                 \
    python${PYTHON_VERSION}-tk                     \
    python${PYTHON_VERSION}-distutils              \
    python3-setuptools

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python${PYTHON_VERSION} get-pip.py

# Force python${PYTHON_VERSION} instead python3.6
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

#-------------------------------------------------------------
#       APEX NN Installation
#-------------------------------------------------------------

WORKDIR ${DEPENDENCIES}

RUN git clone https://github.com/NVIDIA/apex

ENV PYTHONPATH $PYTHONPATH:${DEPENDENCIES}/apex

#-------------------------------------------------------------
#       Other dependencies
#-------------------------------------------------------------

# Deps for python modules
RUN apt-get update -y && \
    apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    ffmpeg

RUN pip3 install opencv-python

#-------------------------------------------------------------
#       NGP_PL Installation
#-------------------------------------------------------------

ENV NGP_PL_3DML="/workspaces/3dml-tools/ML/NeRF/NGP_PL"
ENV PYTHONPATH $PYTHONPATH:$NGP_PL_3DML
ENV MYPYPATH $MYPYPATH:$NGP_PL_3DML

WORKDIR $NGP_PL_3DML

RUN git clone https://github.com/kwea123/ngp_pl.git

COPY . .

#-------------------------------------------------------------
#       Post Processing
#-------------------------------------------------------------

## Switch to non-root user
ENV USER=dock
ENV GROUP=sudo

RUN useradd -ms /bin/bash ${USER} && \
    usermod -aG ${GROUP} ${USER}

# Cleaning Part
USER root
RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*

# Resolve authorization problem
RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" \
    >/etc/sudoers.d/${USER} && \
    chmod 0440 /etc/sudoers.d/${USER}

# Final User
USER ${USER}
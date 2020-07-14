# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt. nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY DAMRSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

FROM nvidia/cuda:10.1-base-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl ca-certificates sudo git bzip2 libx11-6 \
    gcc g++ make cmake zlib1g-dev swig libsm6 libxext6 \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    wget llvm libncurses5-dev xz-utils tk-dev libxrender1\
    libxml2-dev libxmlsec1-dev libffi-dev libcairo2-dev libjpeg-dev libgif-dev chromium-browser

RUN adduser --disabled-password --gecos '' --shell /bin/bash user && chown -R user:user /home/user
RUN echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/90-pyrado
USER user
WORKDIR /home/user

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b \
 && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH /home/user/miniconda3/bin:$PATH

RUN conda update conda \
 && conda update --all

COPY --chown=user:user . SimuRLacra

WORKDIR /home/user/SimuRLacra
RUN bash setup_env.sh
SHELL ["conda", "run", "-n", "pyrado", "/bin/bash", "-c"]

RUN echo "export PATH=/home/user/miniconda3/bin:$PATH" >> ~/.bashrc
RUN echo "conda activate pyrado" >> ~/.bashrc

RUN python setup_deps.py dep_libraries -j4
#RUN python setup_deps.py all --use-cuda -j4

#RUN conda install pytorch torchvision
RUN conda install pytorch cudatoolkit=10.1 -c pytorch

RUN python setup_deps.py separate_pytorch -j4
RUN python setup_deps.py pytorch_based -j4
ENV PATH /opt/conda/envs/pyrado/bin:$PATH
ENV PYTHONPATH /home/user/SimuRLacra/RcsPySim/build/lib:/home/user/SimuRLacra/Pyrado/:$PYTHONPATH
ENV RCSVIEWER_SIMPLEGRAPHICS 1
RUN sudo rm -rf /var/lib/apt/lists/*

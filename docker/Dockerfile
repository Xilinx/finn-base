# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FROM mcr.microsoft.com/azureml/onnxruntime:v1.4.0

LABEL maintainer="Yaman Umuroglu <yamanu@xilinx.com>"

ARG GROUP
ARG GID
ARG USER
ARG UID

ENV SHELL=/bin/bash

# Run below commands as root
USER root

RUN apt-get update
RUN apt-get install -y git build-essential verilator python3-dev
# use python3 and pip3 as python and pip defaults
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

# Copy entrypoint scripts to the workdir (as root)
WORKDIR /usr/local/bin
COPY docker/entrypoint.sh .
COPY docker/run_tests.sh .
COPY docker/build_docs.sh .
RUN chmod +x entrypoint.sh run_tests.sh build_docs.sh

# This creates /workspace if it doesn't exist
WORKDIR /workspace

# Add user
RUN groupadd -o -g ${GID} ${GROUP} && \
    useradd -u ${UID} -g ${GROUP} -ms /bin/bash ${USER} && \
    usermod -aG sudo ${USER} && \
    chown -R ${USER}:${GROUP} /workspace

# Color prompt
RUN echo "PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '" >> /home/${USER}/.bashrc

# Install Python package requirements
COPY docker/requirements.txt .
RUN python -mpip install --upgrade pip && \
    pip install --upgrade setuptools &&\
    pip install -r requirements.txt && \
    rm requirements.txt

# Install custom fork of pyverilator
RUN pip install git+https://github.com/maltanar/pyverilator.git#egg=pyverilator

# Install pytest-xdist (not in requirements, only for faster testing in Docker)
RUN pip install pytest-xdist==2.0.0

# Run below commands as user
USER ${USER}

# Set ENV
ENV PYTHONPATH "${PYTHONPATH}:/workspace/finn-base/src"
ENV PATH "${PATH}:/home/${USER}/.local/bin"

# Set workdir to gray before launching container
WORKDIR /workspace/finn-base

ENTRYPOINT ["entrypoint.sh"]
CMD ["bash"]

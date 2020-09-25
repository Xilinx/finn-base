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
# * Neither the name of finn-base nor the names of its
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

FROM mcr.microsoft.com/azureml/onnxruntime:latest
MAINTAINER Yaman Umuroglu <yamanu@xilinx.com>
ARG GID
ARG GNAME
ARG UNAME
ARG UID

RUN pip install --upgrade pip

# Add host user
RUN groupadd -g $GID $GNAME -f
RUN useradd -u $UID -g $GNAME -ms /bin/bash $UNAME
RUN usermod -aG sudo $UNAME
USER $UNAME

# Set ENV
ENV PYTHONPATH "${PYTHONPATH}:/workspace/finn-base/src"
ENV PATH "${PATH}:/home/$UNAME/.local/bin"

# Color prompt
RUN echo "PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '" >> /home/$UNAME/.bashrc

# Copy entrypoint script as root then switch back to user
USER root
COPY ./docker/entrypoint.sh /usr/local/bin/
COPY ./docker/quicktest.sh /usr/local/bin/
COPY ./docker/docsbuild.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/quicktest.sh
RUN chmod +x /usr/local/bin/docsbuild.sh
USER $UNAME

ENTRYPOINT ["entrypoint.sh"]
CMD ["bash"]

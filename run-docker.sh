#!/bin/bash
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

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'  # No Color

# Green echo
gecho () {
  echo -e "${GREEN}$1${NC}"
}

# Red echo
recho () {
  echo -e "${RED}$1${NC}"
}

DOCKER_GROUP=$(id -gn)
DOCKER_GID=$(id -g)
DOCKER_USER=$(id -un)
DOCKER_UID=$(id -u)
DOCKER_TAG="finn-base"

# Absolute path to this script.
SCRIPT=$(readlink -f "$0")
# Absolute path of dir this script is in.
SCRIPTPATH=$(dirname "${SCRIPT}")

DOCKER_INTERACTIVE=""

if [ "$1" = "tests" ]; then
        echo "Running test suite"
        DOCKER_CMD="run_tests.sh"
elif [ "$1" = "docs" ]; then
        echo "Building docs"
        DOCKER_CMD="build_docs.sh"
elif [ "$1" = "bash" ]; then
        gecho "Running container in interactive mode"
        DOCKER_CMD="bash"
        DOCKER_INTERACTIVE="-it"
else
        recho "Provide one of the following options to run-docker.sh:"
        recho "  {tests, docs, bash}"
        exit -1
fi

# Build the finn-base docker image
docker build -f docker/Dockerfile -t ${DOCKER_TAG} \
            --build-arg GROUP=${DOCKER_GROUP} \
            --build-arg GID=${DOCKER_GID} \
            --build-arg USER=${DOCKER_USER} \
            --build-arg UID=${DOCKER_UID} \
            .

# Launch container with current directory mounted
docker run -t --rm ${DOCKER_INTERACTIVE} \
           -v ${SCRIPTPATH}:/workspace/finn-base \
           ${DOCKER_TAG} ${DOCKER_CMD}

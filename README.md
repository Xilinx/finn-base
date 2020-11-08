## <img src=https://raw.githubusercontent.com/Xilinx/finn/master/docs/img/finn-logo.png width=128/> Core Components for Quantized Neural Network Inference

[![Gitter](https://badges.gitter.im/xilinx-finn/community.svg)](https://gitter.im/xilinx-finn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![ReadTheDocs](https://readthedocs.org/projects/finn-base/badge/?version=latest&style=plastic)](http://finn-base.readthedocs.io/)

`finn-base` is part of the [FINN project](https://xilinx.github.io/finn/) and provides the core infrastructure for the [FINN compiler](https://github.com/Xilinx/finn/), including:

* wrapper around ONNX models for easier manipulation
* infrastructure for applying transformation and analysis passes on ONNX graphs
* infrastructure for defining and executing custom ONNX ops (for verification and code generation)
* extensions to ONNX models using annotations, including few-bit data types, sparsity and data layout specifiers
* several transformation passes, including topological sorting, constant folding and convolution lowering
* several custom ops including im2col and multi-thresholding for quantized activations
* several utility functions, including packing for few-bit integers

## Installation

`finn-base` can be installed via pip by [following these instructions](http://finn-base.readthedocs.io/).

## Documentation

You can view the documentation on [readthedocs](https://finn-base.readthedocs.io) or build them locally using `./run-docker.sh docs`.

## Community

We have a [gitter channel](https://gitter.im/xilinx-finn/community) where you can ask questions. You can use the GitHub issue tracker to report bugs, but please don't file issues to ask questions as this is better handled in the gitter channel.

We also heartily welcome contributions to the project, please check out the [contribution guidelines](CONTRIBUTING.md) and the [list of open issues](https://github.com/Xilinx/finn-base/issues). Don't hesitate to get in touch over [Gitter](https://gitter.im/xilinx-finn/community) to discuss your ideas.

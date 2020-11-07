*********
finn-base
*********

.. note:: **finn-base** is currently under active development. APIs will likely change.

``finn-base`` is part of the `FINN
project <https://xilinx.github.io/finn/>`__ and provides the core
infrastructure for the `FINN
compiler <https://github.com/Xilinx/finn/>`__, including:

-  wrapper around ONNX models for easier manipulation
-  infrastructure for applying transformation and analysis passes on
   ONNX graphs
-  infrastructure for defining and executing custom ONNX ops (for
   verification and code generation)
-  extensions to ONNX models using annotations, including few-bit data
   types, sparsity and data layout specifiers
-  several transformation passes, including topological sorting,
   constant folding and convolution lowering
-  several custom ops including im2col and multi-thresholding for
   quantized activations

.. toctree::
   :maxdepth: 2
   :hidden:

   README <readme>
   API <api/modules>
   License <license>
   Contributors <authors>
   Index <genindex>


* :ref:`modindex`
* :ref:`search`

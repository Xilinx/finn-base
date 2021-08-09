# Copyright (c) 2021 Xilinx, Inc.
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


from onnx import TensorProto, helper

from finn.transformation.base import Transformation
from finn.transformation.general import RemoveUnusedTensors
from finn.util.basic import get_by_name

# ToDo: Similarly to the ops, this should maybe get moved from finn-base into qonnx.
# ToDo: Should these parameters move into a parent class for all NHWC trafos?
# ToDo: I also need some of these parameters in the nhwc op wrappers,
#  so maybe this should get moved to a location, where both, the ops and the trafos
#  can access it.
# Standard ONNX nodes which require a NHWC data format to function properly
_nchw_node_types = ["Conv", "MaxPool", "BatchNormalization"]
_to_chan_last_args = (0, 2, 3, 1)
_to_chan_first_args = (0, 3, 1, 2)

# Nodes, which do not modify the shape of the tensor
# And modify all values in the same way.
_move_through_nodes = ["Quant"]

# Nodes, which do not modify the shape of the tensor,
# And modify all values in the same way, if the second tensor is a scalar.
_move_through_nodes_if_scalar = ["Mul", "Div", "Sub", "Add"]


def applyTrafoAndCheckForChange(model, transformation):
    """
    Applies a transformation and checks if the model changed in any way.
    Returns:
        The transformed model
        Boolean indicating if the model has changed.
    """
    previous_model_string = model.model.SerializeToString()
    model = model.transform(transformation())
    new_model_string = model.model.SerializeToString()
    if previous_model_string == new_model_string:
        model_changed = False
    else:
        model_changed = True

    return model, model_changed


class ConvertToNHWCAndClean(Transformation):
    """
    Converts data layout dependent nodes to NHWC nodes and inserts transformations.
    Then it tries to eliminate as many transformations as possible and moves the
    still existing ones as far upstream as possible.
    ToDo: Implement downstream transformation? It's currently not really needed.
    """

    def apply(self, model):
        model = model.transform(InsertNHWCDomainsAndTrafos())
        max_tries = 100
        for i in range(max_tries):
            # Apply RemoveConsecutiveChanFirstAndChanLastTrafos
            model_changed = False
            model, m_changed = applyTrafoAndCheckForChange(
                model, RemoveConsecutiveChanFirstAndChanLastTrafos
            )
            model_changed |= m_changed

            # Apply MoveChanLastUpstream
            model, m_changed = applyTrafoAndCheckForChange(model, MoveChanLastUpstream)
            model_changed |= m_changed

            # Run RemoveConsecutiveChanFirstAndChanLastTrafos again,
            # if something changed in the previous trafo
            if m_changed:
                model, m_changed = applyTrafoAndCheckForChange(
                    model, RemoveConsecutiveChanFirstAndChanLastTrafos
                )
                model_changed |= m_changed

            # Apply MoveChanLastDownStream
            model, m_changed = applyTrafoAndCheckForChange(
                model, MoveChanFirstDownstream
            )
            model_changed |= m_changed

            # Run RemoveConsecutiveChanFirstAndChanLastTrafos again,
            # if something changed in the previous trafo
            if m_changed:
                model, m_changed = applyTrafoAndCheckForChange(
                    model, RemoveConsecutiveChanFirstAndChanLastTrafos
                )
                model_changed |= m_changed

            # Do some cleanup
            if model_changed:
                model = model.transform(RemoveUnusedTensors())
            else:
                break

        return model, False


class InsertNHWCDomainsAndTrafos(Transformation):
    """Inserts NHWC domain, where required and also inserts required transposes."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # Find nodes, where the domain should be changed
        for n in graph.node:
            node_ind += 1
            if (n.op_type in _nchw_node_types) and (n.domain == ""):
                running_node_index = node_ind
                # Insert transformation nodes for input nodes
                input_nodes = n.input
                # Skip for BatchNorm and 2D input tensors,
                # these contain only channels and need no transpose.
                NCHW_shape = model.get_tensor_shape(input_nodes[0])
                if n.op_type == "BatchNormalization" and len(NCHW_shape) == 2:
                    continue

                for i, inp in enumerate(input_nodes):
                    # Skip higher "order" inputs of the Batch-Norm,
                    # these don't need a transpose.
                    if n.op_type == "BatchNormalization" and i > 0:
                        continue
                    # Get the shape of the input tensor
                    # and convert it to the shape for the intermediate tensor
                    NCHW_shape = model.get_tensor_shape(inp)
                    assert (
                        len(NCHW_shape) == 4
                    ), "NCHW to NHWC conversion is only available for 4D tensors."
                    NHWC_shape = [NCHW_shape[idx] for idx in _to_chan_last_args]
                    # Intermediat tensor
                    inp_trans_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        NHWC_shape,
                    )
                    graph.value_info.append(inp_trans_out)
                    inp_trans_out = inp_trans_out.name

                    # NCHW -> NHWC transpose
                    inp_trans_node = helper.make_node(
                        "Transpose", [inp], [inp_trans_out], perm=_to_chan_last_args
                    )
                    graph.node.insert(running_node_index, inp_trans_node)
                    running_node_index += 1

                    # Attach to original node
                    n.input[i] = inp_trans_out

                # Insert transformation nodes for output nodes
                output_ndes = n.output
                for i, outp in enumerate(output_ndes):
                    NCHW_shape = model.get_tensor_shape(outp)
                    assert (
                        len(NCHW_shape) == 4
                    ), "NCHW to NHWC conversion is only available for 4D tensors."
                    NHWC_shape = [NCHW_shape[idx] for idx in _to_chan_last_args]
                    # Intermediat tensor
                    outp_trans_in = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        NHWC_shape,
                    )
                    graph.value_info.append(outp_trans_in)
                    outp_trans_in = outp_trans_in.name

                    # NCHW -> NHWC transpose
                    outp_trans_node = helper.make_node(
                        "Transpose", [outp_trans_in], [outp], perm=_to_chan_first_args
                    )
                    graph.node.insert(running_node_index, outp_trans_node)
                    running_node_index += 1

                    # Attach to original node
                    n.output[i] = outp_trans_in

                # Modify domain
                n.domain = "qonnx.custom_op.nhwc"
                # Set modified flag
                graph_modified = True

        return model, graph_modified


class RemoveConsecutiveChanFirstAndChanLastTrafos(Transformation):
    """Remove two consecutive transformations, which would do:
    (NHWC -> NCHW) -> (NCHW -> NHWC)
    Or more concrete, the first converts to channels and the scond to channels last.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        # Find fitting transpose node pairs
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose":

                # Check that this is a "to chan first" trafo
                perm_1 = get_by_name(n.attribute, "perm")
                if list(_to_chan_first_args) == perm_1.ints:

                    successor_nodes = model.find_direct_successors(n)
                    assert len(successor_nodes) == 1, (
                        "Transpose nodes should only have one output,"
                        " I don't think more than one would even be possible."
                    )
                    successor_node = successor_nodes[0]

                    if successor_node.op_type == "Transpose":
                        # Check that this is a "to chan last" trafo,
                        # if so both can get removed.
                        perm_2 = get_by_name(successor_node.attribute, "perm")
                        if list(_to_chan_last_args) == perm_2.ints:
                            # Connect original input to new output
                            input_tensor = n.input[0]
                            output_tensor_name = successor_node.output[0]

                            target_nodes = model.find_direct_successors(successor_node)
                            assert len(target_nodes) == 1, (
                                "Transpose nodes should only have one output,"
                                " I don't think more than one would even be possible."
                            )

                            target_node = target_nodes[0]
                            for i, inp in enumerate(target_node.input):
                                if inp == output_tensor_name:
                                    target_node.input[i] = input_tensor

                            # remove old nodes
                            graph.node.remove(n)
                            graph.node.remove(successor_node)

                            graph_modified = True

                            # ToDo: Figure out if the tensors,
                            #  which are now "hanging in the air" must get removed.
        return model, graph_modified


class MoveChanLastUpstream(Transformation):
    """Moves channel last transformations further upstream."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # Find transpose nodes, which are "to chan last" trafos
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose":
                perm = get_by_name(n.attribute, "perm")
                if list(_to_chan_last_args) == perm.ints:
                    predecessors = model.find_direct_predecessors(n)
                    # Check if we reached the top of the graph
                    if predecessors is None:
                        continue
                    assert len(predecessors) == 1, (
                        "Transpose nodes should only have one input, "
                        "I don't think more than one would even be possible."
                    )
                    predecessor = predecessors[0]

                    # Check if we can simply move through the previous node
                    move_through_valid = predecessor.op_type in _move_through_nodes
                    # Check if we have a node, which applies a scalar change,
                    # then we can also move through.
                    if predecessor.op_type in _move_through_nodes_if_scalar:
                        second_inp_shape = model.get_tensor_shape(predecessor.input[1])
                        if second_inp_shape == [1] or second_inp_shape == []:
                            move_through_valid |= True

                    # Apply move through trafo if possible
                    if move_through_valid:
                        # Input tensors are always input 0
                        inp = predecessor.input[0]
                        if isinstance(model.get_initializer(inp), type(None)):
                            # Swap around node "predecessor" and "n"
                            # collect tensors
                            tensor_1 = inp
                            tensor_2 = n.input[0]
                            tensor_3 = n.output[0]
                            # Now connect the tensors to the nodes again,
                            # but in different order
                            n.input[0] = tensor_1
                            n.output[0] = tensor_2
                            predecessor.input[0] = tensor_2
                            predecessor.output[0] = tensor_3

                            # Change the shape of the middle tensor
                            target_shape = model.get_tensor_shape(tensor_3)
                            model.set_tensor_shape(tensor_2, target_shape)

                            graph_modified = True
                        else:
                            # Explicitly apply the transpose to the initializer
                            # of the previous node
                            target_tensor = model.get_initializer(inp)
                            target_tensor = target_tensor.transpose(perm.ints)
                            model.set_initializer(inp, target_tensor)
                            # Reconnect predecessor and delete transpose node
                            predecessor.output[0] = n.output[0]
                            graph.node.remove(n)

                            graph_modified = True
                        return model, graph_modified

        return model, graph_modified


class MoveChanFirstDownstream(Transformation):
    """
    Moves channel first transformations further downstream.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        # Find transpose nodes, which are "to chan first" trafos
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Transpose":
                perm = get_by_name(n.attribute, "perm")
                if list(_to_chan_first_args) == perm.ints:
                    successors = model.find_direct_successors(n)
                    assert (
                        len(successors) == 1
                    ), "Transpose nodes should only have one output"
                    successor = successors[0]

                    # Check if we can simply move through the next node
                    move_through_valid = successor.op_type in _move_through_nodes
                    # Check if we have a node, which applies a scalar change,
                    # then we can also move through.
                    if successor.op_type in _move_through_nodes_if_scalar:
                        second_inp_shape = model.get_tensor_shape(successor.input[1])
                        if second_inp_shape == [1] or second_inp_shape == []:
                            move_through_valid |= True
                    # Apply move through trafo if possible
                    if move_through_valid:
                        # Collect all tensors connecting n and successor
                        # and surrounding nodes
                        tensor_1 = n.input[0]
                        tensor_2 = n.output[0]
                        tensor_3 = successor.output[0]
                        # Now connect the tensors to the nodes again,
                        # but in different order
                        successor.input[0] = tensor_1
                        successor.output[0] = tensor_2
                        n.input[0] = tensor_2
                        n.output[0] = tensor_3

                        # Change the shape of the middle tensor
                        target_shape = model.get_tensor_shape(tensor_1)
                        model.set_tensor_shape(tensor_2, target_shape)

                        graph_modified = True

        return model, graph_modified

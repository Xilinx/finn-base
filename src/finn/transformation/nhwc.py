from onnx import TensorProto, helper

from finn.transformation.base import Transformation
from finn.util.basic import get_by_name

# Standard ONNX nodes which require a nchw data format to function properly
_nchw_node_types = ["Conv", "MaxPool", "BatchNormalization"]
_to_chan_last_args = (0, 2, 3, 1)
_to_chan_first_args = (0, 3, 1, 2)

# Nodes, which do not modify the shape of the tensor, only the values.
_move_through_nodes = ["Quant"]


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
                n.domain = "finn-hls4ml.nhwc"
                # Set modified flag
                graph_modified = True

        return (model, graph_modified)


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
        return (model, graph_modified)


class MoveChanLastUpstream(Transformation):
    """Moves channel last transformations further upstream.
    Currently supported nodes to move along: Quant
    ToDo: Support more nodes, like Mul, Sub, Div.
    """

    # To

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
                    assert len(predecessors) == 1, (
                        "Transpose nodes should only have one input, "
                        "I don't think more than one would even be possible."
                    )
                    predecessor = predecessors[0]

                    # Check if we can simply move through the previous node
                    if predecessor.op_type in _move_through_nodes:
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
                            # Explicitly apply the transpose to the initalizer
                            # of the previous node
                            target_tensor = model.get_initializer(inp)
                            target_tensor = target_tensor.transpose(perm.ints)
                            model.set_initializer(inp, target_tensor)
                            # Reconnect predecessor and delete transpose node
                            predecessor.output[0] = n.output[0]
                            graph.node.remove(n)

                            graph_modified = True

        return (model, graph_modified)

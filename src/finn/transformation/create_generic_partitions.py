# Copyright (c) 2020, Xilinx
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
# * Neither the name of FINN nor the names of its
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

import copy
import pathlib
from onnx import helper

from finn.transformation.base import Transformation
from finn.util.basic import make_build_dir


class PartitionFromLambda(Transformation):
    """Split a graph into partitions based on a node -> partition ID assignment.
    Each resulting partition node has a model attribute indicating the name of
    the associated onnx file."""

    def __init__(self, partitioning=lambda node: -1, partition_dir=None):
        super().__init__()
        self.partitioning = partitioning
        self.partition_dir = partition_dir

    def apply(self, model):
        # partitions may not cover the original graph completely
        # partitions must contain consecutive nodes (based on node index!)
        # assume there is a single input/output to/from each partition
        # TODO: improve support for more complex topologies

        original_nodes = list(model.graph.node)
        partition_assignment = list(map(self.partitioning, original_nodes))

        # check if partitioning obeys the topological order
        partition_ids = []
        for i, partition_id in enumerate(partition_assignment):
            if partition_id > -1:
                if partition_id in partition_ids:
                    assert (
                        partition_id == partition_assignment[i - 1]
                    ), """partitions must contain consecutive nodes"""
                else:
                    partition_ids.append(partition_id)

        # prepare dir for generated .onnx models
        if self.partition_dir is None:
            self.partition_dir = make_build_dir("partitioning_")
        else:
            pathlib.Path(self.partition_dir).mkdir(parents=True, exist_ok=True)

        for partition_id in partition_ids:
            # evaluate partitioning fct. on original graph nodes,
            # not on the model we are currently modifying.
            # -> avoids consistency issues
            # -> allows for partitioning based on node indices
            partition_nodes = list(
                filter(lambda x: self.partitioning(x) == partition_id, original_nodes)
            )

            all_nodes = list(model.graph.node)
            non_partition_nodes = list(
                filter(lambda x: x not in partition_nodes, all_nodes)
            )

            # partition the model into two models
            p_model = copy.deepcopy(model)
            non_p_model = model
            # remove all non-partition nodes from the partition model
            for node_to_remove in non_partition_nodes:
                p_model.graph.node.remove(node_to_remove)
            # identify the entry and exit points for the partition part
            p_in = p_model.graph.node[0].input[0]
            p_out = p_model.graph.node[-1].output[0]
            p_in_vi = p_model.get_tensor_valueinfo(p_in)
            p_out_vi = p_model.get_tensor_valueinfo(p_out)
            # set p graph in/out to be p_in/p_out
            p_model.graph.input.remove(p_model.graph.input[0])
            p_model.graph.input.insert(0, p_in_vi)
            p_model.graph.output.remove(p_model.graph.output[0])
            p_model.graph.output.insert(0, p_out_vi)
            # remove redundant in/out value_info entries
            if p_in_vi in p_model.graph.value_info:
                p_model.graph.value_info.remove(p_in_vi)
            if p_out_vi in p_model.graph.value_info:
                p_model.graph.value_info.remove(p_out_vi)

            # save model
            p_model_filename = (
                self.partition_dir + "/partition_" + str(partition_id) + ".onnx"
            )
            p_model.cleanup()
            p_model.save(p_model_filename)
            # remove all dataflow nodes from the non-dataflow model
            # keep track of where the dataflow part starts
            p_start_ind = all_nodes.index(partition_nodes[0])
            for node_to_remove in partition_nodes:
                non_p_model.graph.node.remove(node_to_remove)
            # create StreamingDataflow node with p_in/p_out io
            p_node = helper.make_node(
                "GenericPartition",
                [p_in],
                [p_out],
                name="GenericPartition_" + str(partition_id),
                # use the model attribute to mark the p model
                model=p_model_filename,
                domain="finn.custom_op.general",
            )
            non_p_model.graph.node.insert(p_start_ind, p_node)
            model = non_p_model

        return (model, False)


class PartitionFromDict(Transformation):
    """Split a graph into partitions based on a dict that assigns node indices to
    integer partition IDs. Each resulting partition node has a model attribute
    indicating the name of the associated onnx file."""

    def __init__(self, partitioning={}, partition_dir=None):
        super().__init__()
        self.partitioning = partitioning
        self.partition_dir = partition_dir

    def apply(self, model):
        # input dict format: { partition_id : node_index_list }
        # example:
        #  {
        #    0 : [3,4,5],
        #    1 : range(10, 15)
        #  }

        # prepare node -> int assignment fct.
        def partitioning_func(node):
            partition_id = -1
            for key in self.partitioning:
                if list(model.graph.node).index(node) in list(self.partitioning[key]):
                    assert (
                        partition_id == -1
                    ), """single node assigned to multiple partitions"""
                    partition_id = key

            return partition_id

        # apply partitioning
        model = model.transform(
            PartitionFromLambda(partitioning_func, self.partition_dir)
        )
        return (model, False)

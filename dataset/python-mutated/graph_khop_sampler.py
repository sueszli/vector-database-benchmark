from paddle import _legacy_C_ops
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode

def graph_khop_sampler(row, colptr, input_nodes, sample_sizes, sorted_eids=None, return_eids=False, name=None):
    if False:
        print('Hello World!')
    '\n\n    Graph Khop Sampler API.\n\n    This API is mainly used in Graph Learning domain, and the main purpose is to\n    provide high performance graph khop sampling method with subgraph reindex step.\n    For example, we get the CSC(Compressed Sparse Column) format of the input graph\n    edges as `row` and `colptr`, so as to covert graph data into a suitable format\n    for sampling. And the `input_nodes` means the nodes we need to sample neighbors,\n    and `sample_sizes` means the number of neighbors and number of layers we want\n    to sample.\n\n    Args:\n        row (Tensor): One of the components of the CSC format of the input graph, and\n                      the shape should be [num_edges, 1] or [num_edges]. The available\n                      data type is int32, int64.\n        colptr (Tensor): One of the components of the CSC format of the input graph,\n                         and the shape should be [num_nodes + 1, 1] or [num_nodes].\n                         The data type should be the same with `row`.\n        input_nodes (Tensor): The input nodes we need to sample neighbors for, and the\n                              data type should be the same with `row`.\n        sample_sizes (list|tuple): The number of neighbors and number of layers we want\n                                   to sample. The data type should be int, and the shape\n                                   should only have one dimension.\n        sorted_eids (Tensor, optional): The sorted edge ids, should not be None when `return_eids`\n                              is True. The shape should be [num_edges, 1], and the data\n                              type should be the same with `row`. Default is None.\n        return_eids (bool, optional): Whether to return the id of the sample edges. Default is False.\n        name (str, optional): Name for the operation (optional, default is None).\n                              For more information, please refer to :ref:`api_guide_Name`.\n\n    Returns:\n        - edge_src (Tensor), The src index of the output edges, also means the first column of\n          the edges. The shape is [num_sample_edges, 1] currently.\n        - edge_dst (Tensor), The dst index of the output edges, also means the second column\n          of the edges. The shape is [num_sample_edges, 1] currently.\n        - sample_index (Tensor), The original id of the input nodes and sampled neighbor nodes.\n        - reindex_nodes (Tensor), The reindex id of the input nodes.\n        - edge_eids (Tensor), Return the id of the sample edges if `return_eids` is True.\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n\n            >>> row = [3, 7, 0, 9, 1, 4, 2, 9, 3, 9, 1, 9, 7]\n            >>> colptr = [0, 2, 4, 5, 6, 7, 9, 11, 11, 13, 13]\n            >>> nodes = [0, 8, 1, 2]\n            >>> sample_sizes = [2, 2]\n            >>> row = paddle.to_tensor(row, dtype="int64")\n            >>> colptr = paddle.to_tensor(colptr, dtype="int64")\n            >>> nodes = paddle.to_tensor(nodes, dtype="int64")\n\n            >>> edge_src, edge_dst, sample_index, reindex_nodes = paddle.incubate.graph_khop_sampler(row, colptr, nodes, sample_sizes, False)\n\n    '
    if in_dynamic_mode():
        if return_eids:
            if sorted_eids is None:
                raise ValueError('`sorted_eid` should not be None if return_eids is True.')
            (edge_src, edge_dst, sample_index, reindex_nodes, edge_eids) = _legacy_C_ops.graph_khop_sampler(row, sorted_eids, colptr, input_nodes, 'sample_sizes', sample_sizes, 'return_eids', True)
            return (edge_src, edge_dst, sample_index, reindex_nodes, edge_eids)
        else:
            (edge_src, edge_dst, sample_index, reindex_nodes, _) = _legacy_C_ops.graph_khop_sampler(row, None, colptr, input_nodes, 'sample_sizes', sample_sizes, 'return_eids', False)
            return (edge_src, edge_dst, sample_index, reindex_nodes)
    check_variable_and_dtype(row, 'Row', ('int32', 'int64'), 'graph_khop_sampler')
    if return_eids:
        if sorted_eids is None:
            raise ValueError('`sorted_eid` should not be None if return_eids is True.')
        check_variable_and_dtype(sorted_eids, 'Eids', ('int32', 'int64'), 'graph_khop_sampler')
    check_variable_and_dtype(colptr, 'Col_Ptr', ('int32', 'int64'), 'graph_khop_sampler')
    check_variable_and_dtype(input_nodes, 'X', ('int32', 'int64'), 'graph_khop_sampler')
    helper = LayerHelper('graph_khop_sampler', **locals())
    edge_src = helper.create_variable_for_type_inference(dtype=row.dtype)
    edge_dst = helper.create_variable_for_type_inference(dtype=row.dtype)
    sample_index = helper.create_variable_for_type_inference(dtype=row.dtype)
    reindex_nodes = helper.create_variable_for_type_inference(dtype=row.dtype)
    edge_eids = helper.create_variable_for_type_inference(dtype=row.dtype)
    helper.append_op(type='graph_khop_sampler', inputs={'Row': row, 'Eids': sorted_eids, 'Col_Ptr': colptr, 'X': input_nodes}, outputs={'Out_Src': edge_src, 'Out_Dst': edge_dst, 'Sample_Index': sample_index, 'Reindex_X': reindex_nodes, 'Out_Eids': edge_eids}, attrs={'sample_sizes': sample_sizes, 'return_eids': return_eids})
    if return_eids:
        return (edge_src, edge_dst, sample_index, reindex_nodes, edge_eids)
    else:
        return (edge_src, edge_dst, sample_index, reindex_nodes)
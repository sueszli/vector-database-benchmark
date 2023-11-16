from paddle import _legacy_C_ops
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode
from paddle.utils import deprecated

@deprecated(since='2.4.0', update_to='paddle.geometric.sample_neighbors', level=1, reason='paddle.incubate.graph_sample_neighbors will be removed in future')
def graph_sample_neighbors(row, colptr, input_nodes, eids=None, perm_buffer=None, sample_size=-1, return_eids=False, flag_perm_buffer=False, name=None):
    if False:
        return 10
    '\n\n    Graph Sample Neighbors API.\n\n    This API is mainly used in Graph Learning domain, and the main purpose is to\n    provide high performance of graph sampling method. For example, we get the\n    CSC(Compressed Sparse Column) format of the input graph edges as `row` and\n    `colptr`, so as to convert graph data into a suitable format for sampling.\n    `input_nodes` means the nodes we need to sample neighbors, and `sample_sizes`\n    means the number of neighbors and number of layers we want to sample.\n\n    Besides, we support fisher-yates sampling in GPU version.\n\n    Args:\n        row (Tensor): One of the components of the CSC format of the input graph, and\n                      the shape should be [num_edges, 1] or [num_edges]. The available\n                      data type is int32, int64.\n        colptr (Tensor): One of the components of the CSC format of the input graph,\n                         and the shape should be [num_nodes + 1, 1] or [num_nodes + 1].\n                         The data type should be the same with `row`.\n        input_nodes (Tensor): The input nodes we need to sample neighbors for, and the\n                              data type should be the same with `row`.\n        eids (Tensor): The eid information of the input graph. If return_eids is True,\n                            then `eids` should not be None. The data type should be the\n                            same with `row`. Default is None.\n        perm_buffer (Tensor): Permutation buffer for fisher-yates sampling. If `flag_perm_buffer`\n                              is True, then `perm_buffer` should not be None. The data type should\n                              be the same with `row`. Default is None.\n        sample_size (int): The number of neighbors we need to sample. Default value is\n                           -1, which means returning all the neighbors of the input nodes.\n        return_eids (bool): Whether to return eid information of sample edges. Default is False.\n        flag_perm_buffer (bool): Using the permutation for fisher-yates sampling in GPU. Default\n                                 value is false, means not using it.\n        name (str, optional): Name for the operation (optional, default is None).\n                              For more information, please refer to :ref:`api_guide_Name`.\n\n    Returns:\n        - out_neighbors (Tensor): The sample neighbors of the input nodes.\n        - out_count (Tensor): The number of sampling neighbors of each input node, and the shape should be the same with `input_nodes`.\n        - out_eids (Tensor): If `return_eids` is True, we will return the eid information of the sample edges.\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n            >>> # edges: (3, 0), (7, 0), (0, 1), (9, 1), (1, 2), (4, 3), (2, 4),\n            >>> #        (9, 5), (3, 5), (9, 6), (1, 6), (9, 8), (7, 8)\n            >>> row = [3, 7, 0, 9, 1, 4, 2, 9, 3, 9, 1, 9, 7]\n            >>> colptr = [0, 2, 4, 5, 6, 7, 9, 11, 11, 13, 13]\n            >>> nodes = [0, 8, 1, 2]\n            >>> sample_size = 2\n            >>> row = paddle.to_tensor(row, dtype="int64")\n            >>> colptr = paddle.to_tensor(colptr, dtype="int64")\n            >>> nodes = paddle.to_tensor(nodes, dtype="int64")\n            >>> out_neighbors, out_count = paddle.incubate.graph_sample_neighbors(\n            ...     row,\n            ...     colptr,\n            ...     nodes,\n            ...     sample_size=sample_size\n            ... )\n\n    '
    if return_eids:
        if eids is None:
            raise ValueError('`eids` should not be None if `return_eids` is True.')
    if flag_perm_buffer:
        if perm_buffer is None:
            raise ValueError('`perm_buffer` should not be None if `flag_perm_buffer`is True.')
    if in_dynamic_mode():
        (out_neighbors, out_count, out_eids) = _legacy_C_ops.graph_sample_neighbors(row, colptr, input_nodes, eids, perm_buffer, 'sample_size', sample_size, 'return_eids', return_eids, 'flag_perm_buffer', flag_perm_buffer)
        if return_eids:
            return (out_neighbors, out_count, out_eids)
        return (out_neighbors, out_count)
    check_variable_and_dtype(row, 'Row', ('int32', 'int64'), 'graph_sample_neighbors')
    check_variable_and_dtype(colptr, 'Col_Ptr', ('int32', 'int64'), 'graph_sample_neighbors')
    check_variable_and_dtype(input_nodes, 'X', ('int32', 'int64'), 'graph_sample_neighbors')
    if return_eids:
        check_variable_and_dtype(eids, 'Eids', ('int32', 'int64'), 'graph_sample_neighbors')
    if flag_perm_buffer:
        check_variable_and_dtype(perm_buffer, 'Perm_Buffer', ('int32', 'int64'), 'graph_sample_neighbors')
    helper = LayerHelper('graph_sample_neighbors', **locals())
    out_neighbors = helper.create_variable_for_type_inference(dtype=row.dtype)
    out_count = helper.create_variable_for_type_inference(dtype=row.dtype)
    out_eids = helper.create_variable_for_type_inference(dtype=row.dtype)
    helper.append_op(type='graph_sample_neighbors', inputs={'Row': row, 'Col_Ptr': colptr, 'X': input_nodes, 'Eids': eids if return_eids else None, 'Perm_Buffer': perm_buffer if flag_perm_buffer else None}, outputs={'Out': out_neighbors, 'Out_Count': out_count, 'Out_Eids': out_eids}, attrs={'sample_size': sample_size, 'return_eids': return_eids, 'flag_perm_buffer': flag_perm_buffer})
    if return_eids:
        return (out_neighbors, out_count, out_eids)
    return (out_neighbors, out_count)
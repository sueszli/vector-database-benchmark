import cupy
import cupyx.scipy.sparse
try:
    import pylibcugraph
    pylibcugraph_available = True
except ModuleNotFoundError:
    pylibcugraph_available = False

def connected_components(csgraph, directed=True, connection='weak', return_labels=True):
    if False:
        for i in range(10):
            print('nop')
    'Analyzes the connected components of a sparse graph\n\n    Args:\n        csgraph (cupy.ndarray of cupyx.scipy.sparse.csr_matrix): The adjacency\n            matrix representing connectivity among nodes.\n        directed (bool): If ``True``, it operates on a directed graph. If\n            ``False``, it operates on an undirected graph.\n        connection (str): ``\'weak\'`` or ``\'strong\'``. For directed graphs, the\n            type of connection to use. Nodes i and j are "strongly" connected\n            only when a path exists both from i to j and from j to i.\n            If ``directed`` is ``False``, this argument is ignored.\n        return_labels (bool): If ``True``, it returns the labels for each of\n            the connected components.\n\n    Returns:\n        tuple of int and cupy.ndarray, or int:\n            If ``return_labels`` == ``True``, returns a tuple ``(n, labels)``,\n            where ``n`` is the number of connected components and ``labels`` is\n            labels of each connected components. Otherwise, returns ``n``.\n\n    .. seealso:: :func:`scipy.sparse.csgraph.connected_components`\n    '
    if not pylibcugraph_available:
        raise RuntimeError('pylibcugraph is not available')
    connection = connection.lower()
    if connection not in ('weak', 'strong'):
        raise ValueError("connection must be 'weak' or 'strong'")
    if not directed:
        connection = 'weak'
    if csgraph.ndim != 2:
        raise ValueError('graph should have two dimensions')
    if not cupyx.scipy.sparse.isspmatrix_csr(csgraph):
        csgraph = cupyx.scipy.sparse.csr_matrix(csgraph)
    (m, m1) = csgraph.shape
    if m != m1:
        raise ValueError('graph should be a square array')
    if csgraph.nnz == 0:
        return (m, cupy.arange(m, dtype=csgraph.indices.dtype))
    if connection == 'strong':
        labels = cupy.empty(m, dtype=csgraph.indices.dtype)
        pylibcugraph.strongly_connected_components(offsets=csgraph.indptr, indices=csgraph.indices, weights=None, num_verts=m, num_edges=csgraph.nnz, labels=labels)
    else:
        csgraph += csgraph.T
        if not cupyx.scipy.sparse.isspmatrix_csr(csgraph):
            csgraph = cupyx.scipy.sparse.csr_matrix(csgraph)
        (_, labels) = pylibcugraph.weakly_connected_components(resource_handle=None, graph=None, indices=csgraph.indices, offsets=csgraph.indptr, weights=None, labels=None, do_expensive_check=False)
    count = cupy.zeros((1,), dtype=csgraph.indices.dtype)
    root_labels = cupy.empty((m,), dtype=csgraph.indices.dtype)
    _cupy_count_components(labels, count, root_labels, size=m)
    n = int(count[0])
    if not return_labels:
        return n
    _cupy_adjust_labels(n, cupy.sort(root_labels[:n]), labels)
    return (n, labels)
_cupy_count_components = cupy.ElementwiseKernel('', 'raw I labels, raw int32 count, raw int32 root_labels', '\n    int j = i;\n    while (j != labels[j]) { j = labels[j]; }\n    if (j != i) {\n        labels[i] = j;\n    } else {\n        int k = atomicAdd(&count[0], 1);\n        root_labels[k] = i;\n    }\n    ', '_cupy_count_components')
_cupy_adjust_labels = cupy.ElementwiseKernel('int32 n_root_labels, raw I root_labels', 'I labels', '\n    int cur_label = labels;\n    int j_min = 0;\n    int j_max = n_root_labels - 1;\n    int j = (j_min + j_max) / 2;\n    while (j_min < j_max) {\n        if (cur_label == root_labels[j]) break;\n        if (cur_label < root_labels[j]) {\n            j_max = j - 1;\n        } else {\n            j_min = j + 1;\n        }\n        j = (j_min + j_max) / 2;\n    }\n    labels = j;\n    ', '_cupy_adjust_labels')
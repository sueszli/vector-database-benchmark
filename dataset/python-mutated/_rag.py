import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require

def _edge_generator_from_csr(csr_matrix):
    if False:
        return 10
    'Yield weighted edge triples for use by NetworkX from a CSR matrix.\n\n    This function is a straight rewrite of\n    `networkx.convert_matrix._csr_gen_triples`. Since that is a private\n    function, it is safer to include our own here.\n\n    Parameters\n    ----------\n    csr_matrix : scipy.sparse.csr_matrix\n        The input matrix. An edge (i, j, w) will be yielded if there is a\n        data value for coordinates (i, j) in the matrix, even if that value\n        is 0.\n\n    Yields\n    ------\n    i, j, w : (int, int, float) tuples\n        Each value `w` in the matrix along with its coordinates (i, j).\n\n    Examples\n    --------\n\n    >>> dense = np.eye(2, dtype=float)\n    >>> csr = sparse.csr_matrix(dense)\n    >>> edges = _edge_generator_from_csr(csr)\n    >>> list(edges)\n    [(0, 0, 1.0), (1, 1, 1.0)]\n    '
    nrows = csr_matrix.shape[0]
    values = csr_matrix.data
    indptr = csr_matrix.indptr
    col_indices = csr_matrix.indices
    for i in range(nrows):
        for j in range(indptr[i], indptr[i + 1]):
            yield (i, col_indices[j], values[j])

def min_weight(graph, src, dst, n):
    if False:
        while True:
            i = 10
    'Callback to handle merging nodes by choosing minimum weight.\n\n    Returns a dictionary with `"weight"` set as either the weight between\n    (`src`, `n`) or (`dst`, `n`) in `graph` or the minimum of the two when\n    both exist.\n\n    Parameters\n    ----------\n    graph : RAG\n        The graph under consideration.\n    src, dst : int\n        The verices in `graph` to be merged.\n    n : int\n        A neighbor of `src` or `dst` or both.\n\n    Returns\n    -------\n    data : dict\n        A dict with the `"weight"` attribute set the weight between\n        (`src`, `n`) or (`dst`, `n`) in `graph` or the minimum of the two when\n        both exist.\n\n    '
    default = {'weight': np.inf}
    w1 = graph[n].get(src, default)['weight']
    w2 = graph[n].get(dst, default)['weight']
    return {'weight': min(w1, w2)}

def _add_edge_filter(values, graph):
    if False:
        return 10
    'Create edge in `graph` between central element of `values` and the rest.\n\n    Add an edge between the middle element in `values` and\n    all other elements of `values` into `graph`.  ``values[len(values) // 2]``\n    is expected to be the central value of the footprint used.\n\n    Parameters\n    ----------\n    values : array\n        The array to process.\n    graph : RAG\n        The graph to add edges in.\n\n    Returns\n    -------\n    0 : float\n        Always returns 0. The return value is required so that `generic_filter`\n        can put it in the output array, but it is ignored by this filter.\n    '
    values = values.astype(int)
    center = values[len(values) // 2]
    for value in values:
        if value != center and (not graph.has_edge(center, value)):
            graph.add_edge(center, value)
    return 0.0

class RAG(nx.Graph):
    """The Region Adjacency Graph (RAG) of an image, subclasses :obj:`networkx.Graph`.

    Parameters
    ----------
    label_image : array of int
        An initial segmentation, with each region labeled as a different
        integer. Every unique value in ``label_image`` will correspond to
        a node in the graph.
    connectivity : int in {1, ..., ``label_image.ndim``}, optional
        The connectivity between pixels in ``label_image``. For a 2D image,
        a connectivity of 1 corresponds to immediate neighbors up, down,
        left, and right, while a connectivity of 2 also includes diagonal
        neighbors. See :func:`scipy.ndimage.generate_binary_structure`.
    data : :obj:`networkx.Graph` specification, optional
        Initial or additional edges to pass to :obj:`networkx.Graph`
        constructor. Valid edge specifications include edge list (list of tuples),
        NumPy arrays, and SciPy sparse matrices.
    **attr : keyword arguments, optional
        Additional attributes to add to the graph.
    """

    def __init__(self, label_image=None, connectivity=1, data=None, **attr):
        if False:
            return 10
        super().__init__(data, **attr)
        if self.number_of_nodes() == 0:
            self.max_id = 0
        else:
            self.max_id = max(self.nodes())
        if label_image is not None:
            fp = ndi.generate_binary_structure(label_image.ndim, connectivity)
            output = np.broadcast_to(1.0, label_image.shape)
            output.setflags(write=True)
            ndi.generic_filter(label_image, function=_add_edge_filter, footprint=fp, mode='nearest', output=output, extra_arguments=(self,))

    def merge_nodes(self, src, dst, weight_func=min_weight, in_place=True, extra_arguments=None, extra_keywords=None):
        if False:
            print('Hello World!')
        'Merge node `src` and `dst`.\n\n        The new combined node is adjacent to all the neighbors of `src`\n        and `dst`. `weight_func` is called to decide the weight of edges\n        incident on the new node.\n\n        Parameters\n        ----------\n        src, dst : int\n            Nodes to be merged.\n        weight_func : callable, optional\n            Function to decide the attributes of edges incident on the new\n            node. For each neighbor `n` for `src` and `dst`, `weight_func` will\n            be called as follows: `weight_func(src, dst, n, *extra_arguments,\n            **extra_keywords)`. `src`, `dst` and `n` are IDs of vertices in the\n            RAG object which is in turn a subclass of :obj:`networkx.Graph`. It is\n            expected to return a dict of attributes of the resulting edge.\n        in_place : bool, optional\n            If set to `True`, the merged node has the id `dst`, else merged\n            node has a new id which is returned.\n        extra_arguments : sequence, optional\n            The sequence of extra positional arguments passed to\n            `weight_func`.\n        extra_keywords : dictionary, optional\n            The dict of keyword arguments passed to the `weight_func`.\n\n        Returns\n        -------\n        id : int\n            The id of the new node.\n\n        Notes\n        -----\n        If `in_place` is `False` the resulting node has a new id, rather than\n        `dst`.\n        '
        if extra_arguments is None:
            extra_arguments = []
        if extra_keywords is None:
            extra_keywords = {}
        src_nbrs = set(self.neighbors(src))
        dst_nbrs = set(self.neighbors(dst))
        neighbors = (src_nbrs | dst_nbrs) - {src, dst}
        if in_place:
            new = dst
        else:
            new = self.next_id()
            self.add_node(new)
        for neighbor in neighbors:
            data = weight_func(self, src, dst, neighbor, *extra_arguments, **extra_keywords)
            self.add_edge(neighbor, new, attr_dict=data)
        self.nodes[new]['labels'] = self.nodes[src]['labels'] + self.nodes[dst]['labels']
        self.remove_node(src)
        if not in_place:
            self.remove_node(dst)
        return new

    def add_node(self, n, attr_dict=None, **attr):
        if False:
            for i in range(10):
                print('nop')
        'Add node `n` while updating the maximum node id.\n\n        .. seealso:: :obj:`networkx.Graph.add_node`.'
        if attr_dict is None:
            attr_dict = attr
        else:
            attr_dict.update(attr)
        super().add_node(n, **attr_dict)
        self.max_id = max(n, self.max_id)

    def add_edge(self, u, v, attr_dict=None, **attr):
        if False:
            while True:
                i = 10
        'Add an edge between `u` and `v` while updating max node id.\n\n        .. seealso:: :obj:`networkx.Graph.add_edge`.'
        if attr_dict is None:
            attr_dict = attr
        else:
            attr_dict.update(attr)
        super().add_edge(u, v, **attr_dict)
        self.max_id = max(u, v, self.max_id)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Copy the graph with its max node id.\n\n        .. seealso:: :obj:`networkx.Graph.copy`.'
        g = super().copy()
        g.max_id = self.max_id
        return g

    def fresh_copy(self):
        if False:
            while True:
                i = 10
        "Return a fresh copy graph with the same data structure.\n\n        A fresh copy has no nodes, edges or graph attributes. It is\n        the same data structure as the current graph. This method is\n        typically used to create an empty version of the graph.\n\n        This is required when subclassing Graph with networkx v2 and\n        does not cause problems for v1. Here is more detail from\n        the network migrating from 1.x to 2.x document::\n\n            With the new GraphViews (SubGraph, ReversedGraph, etc)\n            you can't assume that ``G.__class__()`` will create a new\n            instance of the same graph type as ``G``. In fact, the\n            call signature for ``__class__`` differs depending on\n            whether ``G`` is a view or a base class. For v2.x you\n            should use ``G.fresh_copy()`` to create a null graph of\n            the correct type---ready to fill with nodes and edges.\n\n        "
        return RAG()

    def next_id(self):
        if False:
            while True:
                i = 10
        'Returns the `id` for the new node to be inserted.\n\n        The current implementation returns one more than the maximum `id`.\n\n        Returns\n        -------\n        id : int\n            The `id` of the new node to be inserted.\n        '
        return self.max_id + 1

    def _add_node_silent(self, n):
        if False:
            i = 10
            return i + 15
        'Add node `n` without updating the maximum node id.\n\n        This is a convenience method used internally.\n\n        .. seealso:: :obj:`networkx.Graph.add_node`.'
        super().add_node(n)

def rag_mean_color(image, labels, connectivity=2, mode='distance', sigma=255.0):
    if False:
        print('Hello World!')
    'Compute the Region Adjacency Graph using mean colors.\n\n    Given an image and its initial segmentation, this method constructs the\n    corresponding Region Adjacency Graph (RAG). Each node in the RAG\n    represents a set of pixels within `image` with the same label in `labels`.\n    The weight between two adjacent regions represents how similar or\n    dissimilar two regions are depending on the `mode` parameter.\n\n    Parameters\n    ----------\n    image : ndarray, shape(M, N[, ..., P], 3)\n        Input image.\n    labels : ndarray, shape(M, N[, ..., P])\n        The labelled image. This should have one dimension less than\n        `image`. If `image` has dimensions `(M, N, 3)` `labels` should have\n        dimensions `(M, N)`.\n    connectivity : int, optional\n        Pixels with a squared distance less than `connectivity` from each other\n        are considered adjacent. It can range from 1 to `labels.ndim`. Its\n        behavior is the same as `connectivity` parameter in\n        ``scipy.ndimage.generate_binary_structure``.\n    mode : {\'distance\', \'similarity\'}, optional\n        The strategy to assign edge weights.\n\n            \'distance\' : The weight between two adjacent regions is the\n            :math:`|c_1 - c_2|`, where :math:`c_1` and :math:`c_2` are the mean\n            colors of the two regions. It represents the Euclidean distance in\n            their average color.\n\n            \'similarity\' : The weight between two adjacent is\n            :math:`e^{-d^2/sigma}` where :math:`d=|c_1 - c_2|`, where\n            :math:`c_1` and :math:`c_2` are the mean colors of the two regions.\n            It represents how similar two regions are.\n    sigma : float, optional\n        Used for computation when `mode` is "similarity". It governs how\n        close to each other two colors should be, for their corresponding edge\n        weight to be significant. A very large value of `sigma` could make\n        any two colors behave as though they were similar.\n\n    Returns\n    -------\n    out : RAG\n        The region adjacency graph.\n\n    Examples\n    --------\n    >>> from skimage import data, segmentation, graph\n    >>> img = data.astronaut()\n    >>> labels = segmentation.slic(img)\n    >>> rag = graph.rag_mean_color(img, labels)\n\n    References\n    ----------\n    .. [1] Alain Tremeau and Philippe Colantoni\n           "Regions Adjacency Graph Applied To Color Image Segmentation"\n           :DOI:`10.1109/83.841950`\n    '
    graph = RAG(labels, connectivity=connectivity)
    for n in graph:
        graph.nodes[n].update({'labels': [n], 'pixel count': 0, 'total color': np.array([0, 0, 0], dtype=np.float64)})
    for index in np.ndindex(labels.shape):
        current = labels[index]
        graph.nodes[current]['pixel count'] += 1
        graph.nodes[current]['total color'] += image[index]
    for n in graph:
        graph.nodes[n]['mean color'] = graph.nodes[n]['total color'] / graph.nodes[n]['pixel count']
    for (x, y, d) in graph.edges(data=True):
        diff = graph.nodes[x]['mean color'] - graph.nodes[y]['mean color']
        diff = np.linalg.norm(diff)
        if mode == 'similarity':
            d['weight'] = math.e ** (-diff ** 2 / sigma)
        elif mode == 'distance':
            d['weight'] = diff
        else:
            raise ValueError(f"The mode '{mode}' is not recognised")
    return graph

def rag_boundary(labels, edge_map, connectivity=2):
    if False:
        return 10
    "Comouter RAG based on region boundaries\n\n    Given an image's initial segmentation and its edge map this method\n    constructs the corresponding Region Adjacency Graph (RAG). Each node in the\n    RAG represents a set of pixels within the image with the same label in\n    `labels`. The weight between two adjacent regions is the average value\n    in `edge_map` along their boundary.\n\n    labels : ndarray\n        The labelled image.\n    edge_map : ndarray\n        This should have the same shape as that of `labels`. For all pixels\n        along the boundary between 2 adjacent regions, the average value of the\n        corresponding pixels in `edge_map` is the edge weight between them.\n    connectivity : int, optional\n        Pixels with a squared distance less than `connectivity` from each other\n        are considered adjacent. It can range from 1 to `labels.ndim`. Its\n        behavior is the same as `connectivity` parameter in\n        `scipy.ndimage.generate_binary_structure`.\n\n    Examples\n    --------\n    >>> from skimage import data, segmentation, filters, color, graph\n    >>> img = data.chelsea()\n    >>> labels = segmentation.slic(img)\n    >>> edge_map = filters.sobel(color.rgb2gray(img))\n    >>> rag = graph.rag_boundary(labels, edge_map)\n\n    "
    conn = ndi.generate_binary_structure(labels.ndim, connectivity)
    eroded = ndi.grey_erosion(labels, footprint=conn)
    dilated = ndi.grey_dilation(labels, footprint=conn)
    boundaries0 = eroded != labels
    boundaries1 = dilated != labels
    labels_small = np.concatenate((eroded[boundaries0], labels[boundaries1]))
    labels_large = np.concatenate((labels[boundaries0], dilated[boundaries1]))
    n = np.max(labels_large) + 1
    ones = np.broadcast_to(1.0, labels_small.shape)
    count_matrix = sparse.coo_matrix((ones, (labels_small, labels_large)), dtype=int, shape=(n, n)).tocsr()
    data = np.concatenate((edge_map[boundaries0], edge_map[boundaries1]))
    data_coo = sparse.coo_matrix((data, (labels_small, labels_large)))
    graph_matrix = data_coo.tocsr()
    graph_matrix.data /= count_matrix.data
    rag = RAG()
    rag.add_weighted_edges_from(_edge_generator_from_csr(graph_matrix), weight='weight')
    rag.add_weighted_edges_from(_edge_generator_from_csr(count_matrix), weight='count')
    for n in rag.nodes():
        rag.nodes[n].update({'labels': [n]})
    return rag

@require('matplotlib', '>=3.3')
def show_rag(labels, rag, image, border_color='black', edge_width=1.5, edge_cmap='magma', img_cmap='bone', in_place=True, ax=None):
    if False:
        print('Hello World!')
    "Show a Region Adjacency Graph on an image.\n\n    Given a labelled image and its corresponding RAG, show the nodes and edges\n    of the RAG on the image with the specified colors. Edges are displayed between\n    the centroid of the 2 adjacent regions in the image.\n\n    Parameters\n    ----------\n    labels : ndarray, shape (M, N)\n        The labelled image.\n    rag : RAG\n        The Region Adjacency Graph.\n    image : ndarray, shape (M, N[, 3])\n        Input image. If `colormap` is `None`, the image should be in RGB\n        format.\n    border_color : color spec, optional\n        Color with which the borders between regions are drawn.\n    edge_width : float, optional\n        The thickness with which the RAG edges are drawn.\n    edge_cmap : :py:class:`matplotlib.colors.Colormap`, optional\n        Any matplotlib colormap with which the edges are drawn.\n    img_cmap : :py:class:`matplotlib.colors.Colormap`, optional\n        Any matplotlib colormap with which the image is draw. If set to `None`\n        the image is drawn as it is.\n    in_place : bool, optional\n        If set, the RAG is modified in place. For each node `n` the function\n        will set a new attribute ``rag.nodes[n]['centroid']``.\n    ax : :py:class:`matplotlib.axes.Axes`, optional\n        The axes to draw on. If not specified, new axes are created and drawn\n        on.\n\n    Returns\n    -------\n    lc : :py:class:`matplotlib.collections.LineCollection`\n         A collection of lines that represent the edges of the graph. It can be\n         passed to the :meth:`matplotlib.figure.Figure.colorbar` function.\n\n    Examples\n    --------\n    >>> from skimage import data, segmentation, graph\n    >>> import matplotlib.pyplot as plt\n    >>>\n    >>> img = data.coffee()\n    >>> labels = segmentation.slic(img)\n    >>> g =  graph.rag_mean_color(img, labels)\n    >>> lc = graph.show_rag(labels, g, img)\n    >>> cbar = plt.colorbar(lc)\n    "
    from matplotlib import colors
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    if not in_place:
        rag = rag.copy()
    if ax is None:
        (fig, ax) = plt.subplots()
    out = util.img_as_float(image, force_copy=True)
    if img_cmap is None:
        if image.ndim < 3 or image.shape[2] not in [3, 4]:
            msg = 'If colormap is `None`, an RGB or RGBA image should be given'
            raise ValueError(msg)
        out = image[:, :, :3]
    else:
        img_cmap = plt.get_cmap(img_cmap)
        out = color.rgb2gray(image)
        out = img_cmap(out)[:, :, :3]
    edge_cmap = plt.get_cmap(edge_cmap)
    offset = 1
    map_array = np.arange(labels.max() + 1)
    for (n, d) in rag.nodes(data=True):
        for label in d['labels']:
            map_array[label] = offset
        offset += 1
    rag_labels = map_array[labels]
    regions = measure.regionprops(rag_labels)
    for ((n, data), region) in zip(rag.nodes(data=True), regions):
        data['centroid'] = tuple(map(int, region['centroid']))
    cc = colors.ColorConverter()
    if border_color is not None:
        border_color = cc.to_rgb(border_color)
        out = segmentation.mark_boundaries(out, rag_labels, color=border_color)
    ax.imshow(out)
    lines = [[rag.nodes[n1]['centroid'][::-1], rag.nodes[n2]['centroid'][::-1]] for (n1, n2) in rag.edges()]
    lc = LineCollection(lines, linewidths=edge_width, cmap=edge_cmap)
    edge_weights = [d['weight'] for (x, y, d) in rag.edges(data=True)]
    lc.set_array(np.array(edge_weights))
    ax.add_collection(lc)
    return lc
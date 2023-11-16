"""
====================================
Region adjacency graph (RAG) Merging
====================================

This example constructs a Region Adjacency Graph (RAG) and progressively merges
regions that are similar in color. Merging two adjacent regions produces
a new region with all the pixels from the merged regions. Regions are merged
until no highly similar region pairs remain.

"""
from skimage import data, io, segmentation, color
from skimage import graph
import numpy as np

def _weight_mean_color(graph, src, dst, n):
    if False:
        for i in range(10):
            print('nop')
    'Callback to handle merging nodes by recomputing mean color.\n\n    The method expects that the mean color of `dst` is already computed.\n\n    Parameters\n    ----------\n    graph : RAG\n        The graph under consideration.\n    src, dst : int\n        The vertices in `graph` to be merged.\n    n : int\n        A neighbor of `src` or `dst` or both.\n\n    Returns\n    -------\n    data : dict\n        A dictionary with the `"weight"` attribute set as the absolute\n        difference of the mean color between node `dst` and `n`.\n    '
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src, dst):
    if False:
        for i in range(10):
            print('nop')
    'Callback called before merging two nodes of a mean color distance graph.\n\n    This method computes the mean color of `dst`.\n\n    Parameters\n    ----------\n    graph : RAG\n        The graph under consideration.\n    src, dst : int\n        The vertices in `graph` to be merged.\n    '
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count']
img = data.coffee()
labels = segmentation.slic(img, compactness=30, n_segments=400, start_label=1)
g = graph.rag_mean_color(img, labels)
labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False, in_place_merge=True, merge_func=merge_mean_color, weight_func=_weight_mean_color)
out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
io.imshow(out)
io.show()
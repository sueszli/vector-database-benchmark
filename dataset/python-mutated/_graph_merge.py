import numpy as np
import heapq

def _revalidate_node_edges(rag, node, heap_list):
    if False:
        return 10
    'Handles validation and invalidation of edges incident to a node.\n\n    This function invalidates all existing edges incident on `node` and inserts\n    new items in `heap_list` updated with the valid weights.\n\n    rag : RAG\n        The Region Adjacency Graph.\n    node : int\n        The id of the node whose incident edges are to be validated/invalidated\n        .\n    heap_list : list\n        The list containing the existing heap of edges.\n    '
    for nbr in rag.neighbors(node):
        data = rag[node][nbr]
        try:
            data['heap item'][3] = False
            _invalidate_edge(rag, node, nbr)
        except KeyError:
            pass
        wt = data['weight']
        heap_item = [wt, node, nbr, True]
        data['heap item'] = heap_item
        heapq.heappush(heap_list, heap_item)

def _rename_node(graph, node_id, copy_id):
    if False:
        return 10
    'Rename `node_id` in `graph` to `copy_id`.'
    graph._add_node_silent(copy_id)
    graph.nodes[copy_id].update(graph.nodes[node_id])
    for nbr in graph.neighbors(node_id):
        wt = graph[node_id][nbr]['weight']
        graph.add_edge(nbr, copy_id, {'weight': wt})
    graph.remove_node(node_id)

def _invalidate_edge(graph, n1, n2):
    if False:
        print('Hello World!')
    'Invalidates the edge (n1, n2) in the heap.'
    graph[n1][n2]['heap item'][3] = False

def merge_hierarchical(labels, rag, thresh, rag_copy, in_place_merge, merge_func, weight_func):
    if False:
        return 10
    'Perform hierarchical merging of a RAG.\n\n    Greedily merges the most similar pair of nodes until no edges lower than\n    `thresh` remain.\n\n    Parameters\n    ----------\n    labels : ndarray\n        The array of labels.\n    rag : RAG\n        The Region Adjacency Graph.\n    thresh : float\n        Regions connected by an edge with weight smaller than `thresh` are\n        merged.\n    rag_copy : bool\n        If set, the RAG copied before modifying.\n    in_place_merge : bool\n        If set, the nodes are merged in place. Otherwise, a new node is\n        created for each merge..\n    merge_func : callable\n        This function is called before merging two nodes. For the RAG `graph`\n        while merging `src` and `dst`, it is called as follows\n        ``merge_func(graph, src, dst)``.\n    weight_func : callable\n        The function to compute the new weights of the nodes adjacent to the\n        merged node. This is directly supplied as the argument `weight_func`\n        to `merge_nodes`.\n\n    Returns\n    -------\n    out : ndarray\n        The new labeled array.\n\n    '
    if rag_copy:
        rag = rag.copy()
    edge_heap = []
    for (n1, n2, data) in rag.edges(data=True):
        wt = data['weight']
        heap_item = [wt, n1, n2, True]
        heapq.heappush(edge_heap, heap_item)
        data['heap item'] = heap_item
    while len(edge_heap) > 0 and edge_heap[0][0] < thresh:
        (_, n1, n2, valid) = heapq.heappop(edge_heap)
        if valid:
            for nbr in rag.neighbors(n1):
                _invalidate_edge(rag, n1, nbr)
            for nbr in rag.neighbors(n2):
                _invalidate_edge(rag, n2, nbr)
            if not in_place_merge:
                next_id = rag.next_id()
                _rename_node(rag, n2, next_id)
                (src, dst) = (n1, next_id)
            else:
                (src, dst) = (n1, n2)
            merge_func(rag, src, dst)
            new_id = rag.merge_nodes(src, dst, weight_func)
            _revalidate_node_edges(rag, new_id, edge_heap)
    label_map = np.arange(labels.max() + 1)
    for (ix, (n, d)) in enumerate(rag.nodes(data=True)):
        for label in d['labels']:
            label_map[label] = ix
    return label_map[labels]
"""
altgraph.GraphStat - Functions providing various graph statistics
=================================================================
"""

def degree_dist(graph, limits=(0, 0), bin_num=10, mode='out'):
    if False:
        while True:
            i = 10
    '\n    Computes the degree distribution for a graph.\n\n    Returns a list of tuples where the first element of the tuple is the\n    center of the bin representing a range of degrees and the second element\n    of the tuple are the number of nodes with the degree falling in the range.\n\n    Example::\n\n        ....\n    '
    deg = []
    if mode == 'inc':
        get_deg = graph.inc_degree
    else:
        get_deg = graph.out_degree
    for node in graph:
        deg.append(get_deg(node))
    if not deg:
        return []
    results = _binning(values=deg, limits=limits, bin_num=bin_num)
    return results
_EPS = 1.0 / 2.0 ** 32

def _binning(values, limits=(0, 0), bin_num=10):
    if False:
        while True:
            i = 10
    '\n    Bins data that falls between certain limits, if the limits are (0, 0) the\n    minimum and maximum values are used.\n\n    Returns a list of tuples where the first element of the tuple is the\n    center of the bin and the second element of the tuple are the counts.\n    '
    if limits == (0, 0):
        (min_val, max_val) = (min(values) - _EPS, max(values) + _EPS)
    else:
        (min_val, max_val) = limits
    bin_size = (max_val - min_val) / float(bin_num)
    bins = [0] * bin_num
    for value in values:
        try:
            if value - min_val >= 0:
                index = int((value - min_val) / float(bin_size))
                bins[index] += 1
        except IndexError:
            pass
    result = []
    center = bin_size / 2 + min_val
    for (i, y) in enumerate(bins):
        x = center + bin_size * i
        result.append((x, y))
    return result
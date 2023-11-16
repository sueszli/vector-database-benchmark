import networkx as nx
__all__ = ['s_metric']

@nx._dispatch
def s_metric(G, **kwargs):
    if False:
        i = 10
        return i + 15
    'Returns the s-metric [1]_ of graph.\n\n    The s-metric is defined as the sum of the products ``deg(u) * deg(v)``\n    for every edge ``(u, v)`` in `G`.\n\n    Parameters\n    ----------\n    G : graph\n        The graph used to compute the s-metric.\n    normalized : bool (optional)\n        Normalize the value.\n\n        .. deprecated:: 3.2\n\n           The `normalized` keyword argument is deprecated and will be removed\n           in the future\n\n    Returns\n    -------\n    s : float\n        The s-metric of the graph.\n\n    References\n    ----------\n    .. [1] Lun Li, David Alderson, John C. Doyle, and Walter Willinger,\n           Towards a Theory of Scale-Free Graphs:\n           Definition, Properties, and  Implications (Extended Version), 2005.\n           https://arxiv.org/abs/cond-mat/0501169\n    '
    if kwargs:
        if 'normalized' in kwargs:
            import warnings
            warnings.warn('\n\nThe `normalized` keyword is deprecated and will be removed\nin the future. To silence this warning, remove `normalized`\nwhen calling `s_metric`.\n\nThe value of `normalized` is ignored.', DeprecationWarning, stacklevel=3)
        else:
            raise TypeError(f"s_metric got an unexpected keyword argument '{list(kwargs.keys())[0]}'")
    return float(sum((G.degree(u) * G.degree(v) for (u, v) in G.edges())))
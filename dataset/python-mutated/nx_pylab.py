"""
**********
Matplotlib
**********

Draw networks with matplotlib.

Examples
--------
>>> G = nx.complete_graph(5)
>>> nx.draw(G)

See Also
--------
 - :doc:`matplotlib <matplotlib:index>`
 - :func:`matplotlib.pyplot.scatter`
 - :obj:`matplotlib.patches.FancyArrowPatch`
"""
from numbers import Number
import networkx as nx
from networkx.drawing.layout import circular_layout, kamada_kawai_layout, planar_layout, random_layout, shell_layout, spectral_layout, spring_layout
__all__ = ['draw', 'draw_networkx', 'draw_networkx_nodes', 'draw_networkx_edges', 'draw_networkx_labels', 'draw_networkx_edge_labels', 'draw_circular', 'draw_kamada_kawai', 'draw_random', 'draw_spectral', 'draw_spring', 'draw_planar', 'draw_shell']

def draw(G, pos=None, ax=None, **kwds):
    if False:
        return 10
    'Draw the graph G with Matplotlib.\n\n    Draw the graph as a simple representation with no node\n    labels or edge labels and using the full Matplotlib figure area\n    and no axis labels by default.  See draw_networkx() for more\n    full-featured drawing that allows title, axis labels etc.\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    pos : dictionary, optional\n        A dictionary with nodes as keys and positions as values.\n        If not specified a spring layout positioning will be computed.\n        See :py:mod:`networkx.drawing.layout` for functions that\n        compute node positions.\n\n    ax : Matplotlib Axes object, optional\n        Draw the graph in specified Matplotlib axes.\n\n    kwds : optional keywords\n        See networkx.draw_networkx() for a description of optional keywords.\n\n    Examples\n    --------\n    >>> G = nx.dodecahedral_graph()\n    >>> nx.draw(G)\n    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout\n\n    See Also\n    --------\n    draw_networkx\n    draw_networkx_nodes\n    draw_networkx_edges\n    draw_networkx_labels\n    draw_networkx_edge_labels\n\n    Notes\n    -----\n    This function has the same name as pylab.draw and pyplot.draw\n    so beware when using `from networkx import *`\n\n    since you might overwrite the pylab.draw function.\n\n    With pyplot use\n\n    >>> import matplotlib.pyplot as plt\n    >>> G = nx.dodecahedral_graph()\n    >>> nx.draw(G)  # networkx draw()\n    >>> plt.draw()  # pyplot draw()\n\n    Also see the NetworkX drawing examples at\n    https://networkx.org/documentation/latest/auto_examples/index.html\n    '
    import matplotlib.pyplot as plt
    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor('w')
    if ax is None:
        if cf.axes:
            ax = cf.gca()
        else:
            ax = cf.add_axes((0, 0, 1, 1))
    if 'with_labels' not in kwds:
        kwds['with_labels'] = 'labels' in kwds
    draw_networkx(G, pos=pos, ax=ax, **kwds)
    ax.set_axis_off()
    plt.draw_if_interactive()
    return

def draw_networkx(G, pos=None, arrows=None, with_labels=True, **kwds):
    if False:
        return 10
    'Draw the graph G using Matplotlib.\n\n    Draw the graph with Matplotlib with options for node positions,\n    labeling, titles, and many other drawing features.\n    See draw() for simple drawing without labels or axes.\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    pos : dictionary, optional\n        A dictionary with nodes as keys and positions as values.\n        If not specified a spring layout positioning will be computed.\n        See :py:mod:`networkx.drawing.layout` for functions that\n        compute node positions.\n\n    arrows : bool or None, optional (default=None)\n        If `None`, directed graphs draw arrowheads with\n        `~matplotlib.patches.FancyArrowPatch`, while undirected graphs draw edges\n        via `~matplotlib.collections.LineCollection` for speed.\n        If `True`, draw arrowheads with FancyArrowPatches (bendable and stylish).\n        If `False`, draw edges using LineCollection (linear and fast).\n        For directed graphs, if True draw arrowheads.\n        Note: Arrows will be the same color as edges.\n\n    arrowstyle : str (default=\'-\\|>\' for directed graphs)\n        For directed graphs, choose the style of the arrowsheads.\n        For undirected graphs default to \'-\'\n\n        See `matplotlib.patches.ArrowStyle` for more options.\n\n    arrowsize : int or list (default=10)\n        For directed graphs, choose the size of the arrow head\'s length and\n        width. A list of values can be passed in to assign a different size for arrow head\'s length and width.\n        See `matplotlib.patches.FancyArrowPatch` for attribute `mutation_scale`\n        for more info.\n\n    with_labels :  bool (default=True)\n        Set to True to draw labels on the nodes.\n\n    ax : Matplotlib Axes object, optional\n        Draw the graph in the specified Matplotlib axes.\n\n    nodelist : list (default=list(G))\n        Draw only specified nodes\n\n    edgelist : list (default=list(G.edges()))\n        Draw only specified edges\n\n    node_size : scalar or array (default=300)\n        Size of nodes.  If an array is specified it must be the\n        same length as nodelist.\n\n    node_color : color or array of colors (default=\'#1f78b4\')\n        Node color. Can be a single color or a sequence of colors with the same\n        length as nodelist. Color can be string or rgb (or rgba) tuple of\n        floats from 0-1. If numeric values are specified they will be\n        mapped to colors using the cmap and vmin,vmax parameters. See\n        matplotlib.scatter for more details.\n\n    node_shape :  string (default=\'o\')\n        The shape of the node.  Specification is as matplotlib.scatter\n        marker, one of \'so^>v<dph8\'.\n\n    alpha : float or None (default=None)\n        The node and edge transparency\n\n    cmap : Matplotlib colormap, optional\n        Colormap for mapping intensities of nodes\n\n    vmin,vmax : float, optional\n        Minimum and maximum for node colormap scaling\n\n    linewidths : scalar or sequence (default=1.0)\n        Line width of symbol border\n\n    width : float or array of floats (default=1.0)\n        Line width of edges\n\n    edge_color : color or array of colors (default=\'k\')\n        Edge color. Can be a single color or a sequence of colors with the same\n        length as edgelist. Color can be string or rgb (or rgba) tuple of\n        floats from 0-1. If numeric values are specified they will be\n        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.\n\n    edge_cmap : Matplotlib colormap, optional\n        Colormap for mapping intensities of edges\n\n    edge_vmin,edge_vmax : floats, optional\n        Minimum and maximum for edge colormap scaling\n\n    style : string (default=solid line)\n        Edge line style e.g.: \'-\', \'--\', \'-.\', \':\'\n        or words like \'solid\' or \'dashed\'.\n        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)\n\n    labels : dictionary (default=None)\n        Node labels in a dictionary of text labels keyed by node\n\n    font_size : int (default=12 for nodes, 10 for edges)\n        Font size for text labels\n\n    font_color : color (default=\'k\' black)\n        Font color string. Color can be string or rgb (or rgba) tuple of\n        floats from 0-1.\n\n    font_weight : string (default=\'normal\')\n        Font weight\n\n    font_family : string (default=\'sans-serif\')\n        Font family\n\n    label : string, optional\n        Label for graph legend\n\n    kwds : optional keywords\n        See networkx.draw_networkx_nodes(), networkx.draw_networkx_edges(), and\n        networkx.draw_networkx_labels() for a description of optional keywords.\n\n    Notes\n    -----\n    For directed graphs, arrows  are drawn at the head end.  Arrows can be\n    turned off with keyword arrows=False.\n\n    Examples\n    --------\n    >>> G = nx.dodecahedral_graph()\n    >>> nx.draw(G)\n    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout\n\n    >>> import matplotlib.pyplot as plt\n    >>> limits = plt.axis("off")  # turn off axis\n\n    Also see the NetworkX drawing examples at\n    https://networkx.org/documentation/latest/auto_examples/index.html\n\n    See Also\n    --------\n    draw\n    draw_networkx_nodes\n    draw_networkx_edges\n    draw_networkx_labels\n    draw_networkx_edge_labels\n    '
    from inspect import signature
    import matplotlib.pyplot as plt
    valid_node_kwds = signature(draw_networkx_nodes).parameters.keys()
    valid_edge_kwds = signature(draw_networkx_edges).parameters.keys()
    valid_label_kwds = signature(draw_networkx_labels).parameters.keys()
    valid_kwds = (valid_node_kwds | valid_edge_kwds | valid_label_kwds) - {'G', 'pos', 'arrows', 'with_labels'}
    if any((k not in valid_kwds for k in kwds)):
        invalid_args = ', '.join([k for k in kwds if k not in valid_kwds])
        raise ValueError(f'Received invalid argument(s): {invalid_args}')
    node_kwds = {k: v for (k, v) in kwds.items() if k in valid_node_kwds}
    edge_kwds = {k: v for (k, v) in kwds.items() if k in valid_edge_kwds}
    label_kwds = {k: v for (k, v) in kwds.items() if k in valid_label_kwds}
    if pos is None:
        pos = nx.drawing.spring_layout(G)
    draw_networkx_nodes(G, pos, **node_kwds)
    draw_networkx_edges(G, pos, arrows=arrows, **edge_kwds)
    if with_labels:
        draw_networkx_labels(G, pos, **label_kwds)
    plt.draw_if_interactive()

def draw_networkx_nodes(G, pos, nodelist=None, node_size=300, node_color='#1f78b4', node_shape='o', alpha=None, cmap=None, vmin=None, vmax=None, ax=None, linewidths=None, edgecolors=None, label=None, margins=None):
    if False:
        print('Hello World!')
    "Draw the nodes of the graph G.\n\n    This draws only the nodes of the graph G.\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    pos : dictionary\n        A dictionary with nodes as keys and positions as values.\n        Positions should be sequences of length 2.\n\n    ax : Matplotlib Axes object, optional\n        Draw the graph in the specified Matplotlib axes.\n\n    nodelist : list (default list(G))\n        Draw only specified nodes\n\n    node_size : scalar or array (default=300)\n        Size of nodes.  If an array it must be the same length as nodelist.\n\n    node_color : color or array of colors (default='#1f78b4')\n        Node color. Can be a single color or a sequence of colors with the same\n        length as nodelist. Color can be string or rgb (or rgba) tuple of\n        floats from 0-1. If numeric values are specified they will be\n        mapped to colors using the cmap and vmin,vmax parameters. See\n        matplotlib.scatter for more details.\n\n    node_shape :  string (default='o')\n        The shape of the node.  Specification is as matplotlib.scatter\n        marker, one of 'so^>v<dph8'.\n\n    alpha : float or array of floats (default=None)\n        The node transparency.  This can be a single alpha value,\n        in which case it will be applied to all the nodes of color. Otherwise,\n        if it is an array, the elements of alpha will be applied to the colors\n        in order (cycling through alpha multiple times if necessary).\n\n    cmap : Matplotlib colormap (default=None)\n        Colormap for mapping intensities of nodes\n\n    vmin,vmax : floats or None (default=None)\n        Minimum and maximum for node colormap scaling\n\n    linewidths : [None | scalar | sequence] (default=1.0)\n        Line width of symbol border\n\n    edgecolors : [None | scalar | sequence] (default = node_color)\n        Colors of node borders. Can be a single color or a sequence of colors with the\n        same length as nodelist. Color can be string or rgb (or rgba) tuple of floats\n        from 0-1. If numeric values are specified they will be mapped to colors\n        using the cmap and vmin,vmax parameters. See `~matplotlib.pyplot.scatter` for more details.\n\n    label : [None | string]\n        Label for legend\n\n    margins : float or 2-tuple, optional\n        Sets the padding for axis autoscaling. Increase margin to prevent\n        clipping for nodes that are near the edges of an image. Values should\n        be in the range ``[0, 1]``. See :meth:`matplotlib.axes.Axes.margins`\n        for details. The default is `None`, which uses the Matplotlib default.\n\n    Returns\n    -------\n    matplotlib.collections.PathCollection\n        `PathCollection` of the nodes.\n\n    Examples\n    --------\n    >>> G = nx.dodecahedral_graph()\n    >>> nodes = nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))\n\n    Also see the NetworkX drawing examples at\n    https://networkx.org/documentation/latest/auto_examples/index.html\n\n    See Also\n    --------\n    draw\n    draw_networkx\n    draw_networkx_edges\n    draw_networkx_labels\n    draw_networkx_edge_labels\n    "
    from collections.abc import Iterable
    import matplotlib as mpl
    import matplotlib.collections
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        ax = plt.gca()
    if nodelist is None:
        nodelist = list(G)
    if len(nodelist) == 0:
        return mpl.collections.PathCollection(None)
    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as err:
        raise nx.NetworkXError(f'Node {err} has no position.') from err
    if isinstance(alpha, Iterable):
        node_color = apply_alpha(node_color, alpha, nodelist, cmap, vmin, vmax)
        alpha = None
    node_collection = ax.scatter(xy[:, 0], xy[:, 1], s=node_size, c=node_color, marker=node_shape, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, edgecolors=edgecolors, label=label)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    if margins is not None:
        if isinstance(margins, Iterable):
            ax.margins(*margins)
        else:
            ax.margins(margins)
    node_collection.set_zorder(2)
    return node_collection

def draw_networkx_edges(G, pos, edgelist=None, width=1.0, edge_color='k', style='solid', alpha=None, arrowstyle=None, arrowsize=10, edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None, arrows=None, label=None, node_size=300, nodelist=None, node_shape='o', connectionstyle='arc3', min_source_margin=0, min_target_margin=0):
    if False:
        for i in range(10):
            print('nop')
    'Draw the edges of the graph G.\n\n    This draws only the edges of the graph G.\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    pos : dictionary\n        A dictionary with nodes as keys and positions as values.\n        Positions should be sequences of length 2.\n\n    edgelist : collection of edge tuples (default=G.edges())\n        Draw only specified edges\n\n    width : float or array of floats (default=1.0)\n        Line width of edges\n\n    edge_color : color or array of colors (default=\'k\')\n        Edge color. Can be a single color or a sequence of colors with the same\n        length as edgelist. Color can be string or rgb (or rgba) tuple of\n        floats from 0-1. If numeric values are specified they will be\n        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.\n\n    style : string or array of strings (default=\'solid\')\n        Edge line style e.g.: \'-\', \'--\', \'-.\', \':\'\n        or words like \'solid\' or \'dashed\'.\n        Can be a single style or a sequence of styles with the same\n        length as the edge list.\n        If less styles than edges are given the styles will cycle.\n        If more styles than edges are given the styles will be used sequentially\n        and not be exhausted.\n        Also, `(offset, onoffseq)` tuples can be used as style instead of a strings.\n        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)\n\n    alpha : float or array of floats (default=None)\n        The edge transparency.  This can be a single alpha value,\n        in which case it will be applied to all specified edges. Otherwise,\n        if it is an array, the elements of alpha will be applied to the colors\n        in order (cycling through alpha multiple times if necessary).\n\n    edge_cmap : Matplotlib colormap, optional\n        Colormap for mapping intensities of edges\n\n    edge_vmin,edge_vmax : floats, optional\n        Minimum and maximum for edge colormap scaling\n\n    ax : Matplotlib Axes object, optional\n        Draw the graph in the specified Matplotlib axes.\n\n    arrows : bool or None, optional (default=None)\n        If `None`, directed graphs draw arrowheads with\n        `~matplotlib.patches.FancyArrowPatch`, while undirected graphs draw edges\n        via `~matplotlib.collections.LineCollection` for speed.\n        If `True`, draw arrowheads with FancyArrowPatches (bendable and stylish).\n        If `False`, draw edges using LineCollection (linear and fast).\n\n        Note: Arrowheads will be the same color as edges.\n\n    arrowstyle : str (default=\'-\\|>\' for directed graphs)\n        For directed graphs and `arrows==True` defaults to \'-\\|>\',\n        For undirected graphs default to \'-\'.\n\n        See `matplotlib.patches.ArrowStyle` for more options.\n\n    arrowsize : int (default=10)\n        For directed graphs, choose the size of the arrow head\'s length and\n        width. See `matplotlib.patches.FancyArrowPatch` for attribute\n        `mutation_scale` for more info.\n\n    connectionstyle : string (default="arc3")\n        Pass the connectionstyle parameter to create curved arc of rounding\n        radius rad. For example, connectionstyle=\'arc3,rad=0.2\'.\n        See `matplotlib.patches.ConnectionStyle` and\n        `matplotlib.patches.FancyArrowPatch` for more info.\n\n    node_size : scalar or array (default=300)\n        Size of nodes. Though the nodes are not drawn with this function, the\n        node size is used in determining edge positioning.\n\n    nodelist : list, optional (default=G.nodes())\n       This provides the node order for the `node_size` array (if it is an array).\n\n    node_shape :  string (default=\'o\')\n        The marker used for nodes, used in determining edge positioning.\n        Specification is as a `matplotlib.markers` marker, e.g. one of \'so^>v<dph8\'.\n\n    label : None or string\n        Label for legend\n\n    min_source_margin : int (default=0)\n        The minimum margin (gap) at the beginning of the edge at the source.\n\n    min_target_margin : int (default=0)\n        The minimum margin (gap) at the end of the edge at the target.\n\n    Returns\n    -------\n     matplotlib.collections.LineCollection or a list of matplotlib.patches.FancyArrowPatch\n        If ``arrows=True``, a list of FancyArrowPatches is returned.\n        If ``arrows=False``, a LineCollection is returned.\n        If ``arrows=None`` (the default), then a LineCollection is returned if\n        `G` is undirected, otherwise returns a list of FancyArrowPatches.\n\n    Notes\n    -----\n    For directed graphs, arrows are drawn at the head end.  Arrows can be\n    turned off with keyword arrows=False or by passing an arrowstyle without\n    an arrow on the end.\n\n    Be sure to include `node_size` as a keyword argument; arrows are\n    drawn considering the size of nodes.\n\n    Self-loops are always drawn with `~matplotlib.patches.FancyArrowPatch`\n    regardless of the value of `arrows` or whether `G` is directed.\n    When ``arrows=False`` or ``arrows=None`` and `G` is undirected, the\n    FancyArrowPatches corresponding to the self-loops are not explicitly\n    returned. They should instead be accessed via the ``Axes.patches``\n    attribute (see examples).\n\n    Examples\n    --------\n    >>> G = nx.dodecahedral_graph()\n    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))\n\n    >>> G = nx.DiGraph()\n    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])\n    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))\n    >>> alphas = [0.3, 0.4, 0.5]\n    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs\n    ...     arc.set_alpha(alphas[i])\n\n    The FancyArrowPatches corresponding to self-loops are not always\n    returned, but can always be accessed via the ``patches`` attribute of the\n    `matplotlib.Axes` object.\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> G = nx.Graph([(0, 1), (0, 0)])  # Self-loop at node 0\n    >>> edge_collection = nx.draw_networkx_edges(G, pos=nx.circular_layout(G), ax=ax)\n    >>> self_loop_fap = ax.patches[0]\n\n    Also see the NetworkX drawing examples at\n    https://networkx.org/documentation/latest/auto_examples/index.html\n\n    See Also\n    --------\n    draw\n    draw_networkx\n    draw_networkx_nodes\n    draw_networkx_labels\n    draw_networkx_edge_labels\n\n    '
    import matplotlib as mpl
    import matplotlib.collections
    import matplotlib.colors
    import matplotlib.patches
    import matplotlib.path
    import matplotlib.pyplot as plt
    import numpy as np
    use_linecollection = not G.is_directed()
    if arrows in (True, False):
        use_linecollection = not arrows
    if use_linecollection and any([arrowstyle is not None, arrowsize != 10, connectionstyle != 'arc3', min_source_margin != 0, min_target_margin != 0]):
        import warnings
        msg = '\n\nThe {0} keyword argument is not applicable when drawing edges\nwith LineCollection.\n\nTo make this warning go away, either specify `arrows=True` to\nforce FancyArrowPatches or use the default value for {0}.\nNote that using FancyArrowPatches may be slow for large graphs.\n'
        if arrowstyle is not None:
            msg = msg.format('arrowstyle')
        if arrowsize != 10:
            msg = msg.format('arrowsize')
        if connectionstyle != 'arc3':
            msg = msg.format('connectionstyle')
        if min_source_margin != 0:
            msg = msg.format('min_source_margin')
        if min_target_margin != 0:
            msg = msg.format('min_target_margin')
        warnings.warn(msg, category=UserWarning, stacklevel=2)
    if arrowstyle == None:
        if G.is_directed():
            arrowstyle = '-|>'
        else:
            arrowstyle = '-'
    if ax is None:
        ax = plt.gca()
    if edgelist is None:
        edgelist = list(G.edges())
    if len(edgelist) == 0:
        return []
    if nodelist is None:
        nodelist = list(G.nodes())
    if edge_color is None:
        edge_color = 'k'
    edgelist_tuple = list(map(tuple, edgelist))
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
    if np.iterable(edge_color) and len(edge_color) == len(edge_pos) and np.all([isinstance(c, Number) for c in edge_color]):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    def _draw_networkx_edges_line_collection():
        if False:
            return 10
        edge_collection = mpl.collections.LineCollection(edge_pos, colors=edge_color, linewidths=width, antialiaseds=(1,), linestyle=style, alpha=alpha)
        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)
        edge_collection.set_zorder(1)
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)
        return edge_collection

    def _draw_networkx_edges_fancy_arrow_patch():
        if False:
            return 10

        def to_marker_edge(marker_size, marker):
            if False:
                for i in range(10):
                    print('nop')
            if marker in 's^>v<d':
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2
        arrow_collection = []
        if isinstance(arrowsize, list):
            if len(arrowsize) != len(edge_pos):
                raise ValueError('arrowsize should have the same length as edgelist')
        else:
            mutation_scale = arrowsize
        base_connection_style = mpl.patches.ConnectionStyle(connectionstyle)
        max_nodesize = np.array(node_size).max()

        def _connectionstyle(posA, posB, *args, **kwargs):
            if False:
                return 10
            if np.all(posA == posB):
                selfloop_ht = 0.005 * max_nodesize if h == 0 else h
                data_loc = ax.transData.inverted().transform(posA)
                v_shift = 0.1 * selfloop_ht
                h_shift = v_shift * 0.5
                path = [data_loc + np.asarray([0, v_shift]), data_loc + np.asarray([h_shift, v_shift]), data_loc + np.asarray([h_shift, 0]), data_loc, data_loc + np.asarray([-h_shift, 0]), data_loc + np.asarray([-h_shift, v_shift]), data_loc + np.asarray([0, v_shift])]
                ret = mpl.path.Path(ax.transData.transform(path), [1, 4, 4, 4, 4, 4, 4])
            else:
                ret = base_connection_style(posA, posB, *args, **kwargs)
            return ret
        arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
        for (i, (src, dst)) in zip(fancy_edges_indices, edge_pos):
            (x1, y1) = src
            (x2, y2) = dst
            shrink_source = 0
            shrink_target = 0
            if isinstance(arrowsize, list):
                mutation_scale = arrowsize[i]
            if np.iterable(node_size):
                (source, target) = edgelist[i][:2]
                source_node_size = node_size[nodelist.index(source)]
                target_node_size = node_size[nodelist.index(target)]
                shrink_source = to_marker_edge(source_node_size, node_shape)
                shrink_target = to_marker_edge(target_node_size, node_shape)
            else:
                shrink_source = shrink_target = to_marker_edge(node_size, node_shape)
            if shrink_source < min_source_margin:
                shrink_source = min_source_margin
            if shrink_target < min_target_margin:
                shrink_target = min_target_margin
            if len(arrow_colors) > i:
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:
                arrow_color = arrow_colors[i % len(arrow_colors)]
            if np.iterable(width):
                if len(width) > i:
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width
            if np.iterable(style) and (not isinstance(style, str)) and (not isinstance(style, tuple)):
                if len(style) > i:
                    linestyle = style[i]
                else:
                    linestyle = style[i % len(style)]
            else:
                linestyle = style
            arrow = mpl.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=arrowstyle, shrinkA=shrink_source, shrinkB=shrink_target, mutation_scale=mutation_scale, color=arrow_color, linewidth=line_width, connectionstyle=_connectionstyle, linestyle=linestyle, zorder=1)
            arrow_collection.append(arrow)
            ax.add_patch(arrow)
        return arrow_collection
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    w = maxx - minx
    h = maxy - miny
    if use_linecollection:
        edge_viz_obj = _draw_networkx_edges_line_collection()
        selfloops_to_draw = [loop for loop in nx.selfloop_edges(G) if loop in edgelist]
        if selfloops_to_draw:
            fancy_edges_indices = [edgelist_tuple.index(loop) for loop in selfloops_to_draw]
            edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in selfloops_to_draw])
            arrowstyle = '-'
            _draw_networkx_edges_fancy_arrow_patch()
    else:
        fancy_edges_indices = range(len(edgelist))
        edge_viz_obj = _draw_networkx_edges_fancy_arrow_patch()
    (padx, pady) = (0.05 * w, 0.05 * h)
    corners = ((minx - padx, miny - pady), (maxx + padx, maxy + pady))
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return edge_viz_obj

def draw_networkx_labels(G, pos, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, clip_on=True):
    if False:
        i = 10
        return i + 15
    "Draw node labels on the graph G.\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    pos : dictionary\n        A dictionary with nodes as keys and positions as values.\n        Positions should be sequences of length 2.\n\n    labels : dictionary (default={n: n for n in G})\n        Node labels in a dictionary of text labels keyed by node.\n        Node-keys in labels should appear as keys in `pos`.\n        If needed use: `{n:lab for n,lab in labels.items() if n in pos}`\n\n    font_size : int (default=12)\n        Font size for text labels\n\n    font_color : color (default='k' black)\n        Font color string. Color can be string or rgb (or rgba) tuple of\n        floats from 0-1.\n\n    font_weight : string (default='normal')\n        Font weight\n\n    font_family : string (default='sans-serif')\n        Font family\n\n    alpha : float or None (default=None)\n        The text transparency\n\n    bbox : Matplotlib bbox, (default is Matplotlib's ax.text default)\n        Specify text box properties (e.g. shape, color etc.) for node labels.\n\n    horizontalalignment : string (default='center')\n        Horizontal alignment {'center', 'right', 'left'}\n\n    verticalalignment : string (default='center')\n        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}\n\n    ax : Matplotlib Axes object, optional\n        Draw the graph in the specified Matplotlib axes.\n\n    clip_on : bool (default=True)\n        Turn on clipping of node labels at axis boundaries\n\n    Returns\n    -------\n    dict\n        `dict` of labels keyed on the nodes\n\n    Examples\n    --------\n    >>> G = nx.dodecahedral_graph()\n    >>> labels = nx.draw_networkx_labels(G, pos=nx.spring_layout(G))\n\n    Also see the NetworkX drawing examples at\n    https://networkx.org/documentation/latest/auto_examples/index.html\n\n    See Also\n    --------\n    draw\n    draw_networkx\n    draw_networkx_nodes\n    draw_networkx_edges\n    draw_networkx_edge_labels\n    "
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = {n: n for n in G.nodes()}
    text_items = {}
    for (n, label) in labels.items():
        (x, y) = pos[n]
        if not isinstance(label, str):
            label = str(label)
        t = ax.text(x, y, label, size=font_size, color=font_color, family=font_family, weight=font_weight, alpha=alpha, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, transform=ax.transData, bbox=bbox, clip_on=clip_on)
        text_items[n] = t
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return text_items

def draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, rotate=True, clip_on=True):
    if False:
        while True:
            i = 10
    "Draw edge labels.\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    pos : dictionary\n        A dictionary with nodes as keys and positions as values.\n        Positions should be sequences of length 2.\n\n    edge_labels : dictionary (default=None)\n        Edge labels in a dictionary of labels keyed by edge two-tuple.\n        Only labels for the keys in the dictionary are drawn.\n\n    label_pos : float (default=0.5)\n        Position of edge label along edge (0=head, 0.5=center, 1=tail)\n\n    font_size : int (default=10)\n        Font size for text labels\n\n    font_color : color (default='k' black)\n        Font color string. Color can be string or rgb (or rgba) tuple of\n        floats from 0-1.\n\n    font_weight : string (default='normal')\n        Font weight\n\n    font_family : string (default='sans-serif')\n        Font family\n\n    alpha : float or None (default=None)\n        The text transparency\n\n    bbox : Matplotlib bbox, optional\n        Specify text box properties (e.g. shape, color etc.) for edge labels.\n        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.\n\n    horizontalalignment : string (default='center')\n        Horizontal alignment {'center', 'right', 'left'}\n\n    verticalalignment : string (default='center')\n        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}\n\n    ax : Matplotlib Axes object, optional\n        Draw the graph in the specified Matplotlib axes.\n\n    rotate : bool (default=True)\n        Rotate edge labels to lie parallel to edges\n\n    clip_on : bool (default=True)\n        Turn on clipping of edge labels at axis boundaries\n\n    Returns\n    -------\n    dict\n        `dict` of labels keyed by edge\n\n    Examples\n    --------\n    >>> G = nx.dodecahedral_graph()\n    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))\n\n    Also see the NetworkX drawing examples at\n    https://networkx.org/documentation/latest/auto_examples/index.html\n\n    See Also\n    --------\n    draw\n    draw_networkx\n    draw_networkx_nodes\n    draw_networkx_edges\n    draw_networkx_labels\n    "
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for (u, v, d) in G.edges(data=True)}
    else:
        labels = edge_labels
        try:
            (u, v) = next(iter(labels))
        except ValueError as err:
            raise nx.NetworkXError('draw_networkx_edge_labels does not support multiedges.') from err
        except StopIteration:
            pass
    text_items = {}
    for ((n1, n2), label) in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (x1 * label_pos + x2 * (1.0 - label_pos), y1 * label_pos + y2 * (1.0 - label_pos))
        if rotate:
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(np.array((angle,)), xy.reshape((1, 2)))[0]
        else:
            trans_angle = 0.0
        if bbox is None:
            bbox = {'boxstyle': 'round', 'ec': (1.0, 1.0, 1.0), 'fc': (1.0, 1.0, 1.0)}
        if not isinstance(label, str):
            label = str(label)
        t = ax.text(x, y, label, size=font_size, color=font_color, family=font_family, weight=font_weight, alpha=alpha, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, rotation=trans_angle, transform=ax.transData, bbox=bbox, zorder=1, clip_on=clip_on)
        text_items[n1, n2] = t
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    return text_items

def draw_circular(G, **kwargs):
    if False:
        i = 10
        return i + 15
    'Draw the graph `G` with a circular layout.\n\n    This is a convenience function equivalent to::\n\n        nx.draw(G, pos=nx.circular_layout(G), **kwargs)\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    kwargs : optional keywords\n        See `draw_networkx` for a description of optional keywords.\n\n    Notes\n    -----\n    The layout is computed each time this function is called. For\n    repeated drawing it is much more efficient to call\n    `~networkx.drawing.layout.circular_layout` directly and reuse the result::\n\n        >>> G = nx.complete_graph(5)\n        >>> pos = nx.circular_layout(G)\n        >>> nx.draw(G, pos=pos)  # Draw the original graph\n        >>> # Draw a subgraph, reusing the same node positions\n        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> nx.draw_circular(G)\n\n    See Also\n    --------\n    :func:`~networkx.drawing.layout.circular_layout`\n    '
    draw(G, circular_layout(G), **kwargs)

def draw_kamada_kawai(G, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Draw the graph `G` with a Kamada-Kawai force-directed layout.\n\n    This is a convenience function equivalent to::\n\n        nx.draw(G, pos=nx.kamada_kawai_layout(G), **kwargs)\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    kwargs : optional keywords\n        See `draw_networkx` for a description of optional keywords.\n\n    Notes\n    -----\n    The layout is computed each time this function is called.\n    For repeated drawing it is much more efficient to call\n    `~networkx.drawing.layout.kamada_kawai_layout` directly and reuse the\n    result::\n\n        >>> G = nx.complete_graph(5)\n        >>> pos = nx.kamada_kawai_layout(G)\n        >>> nx.draw(G, pos=pos)  # Draw the original graph\n        >>> # Draw a subgraph, reusing the same node positions\n        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> nx.draw_kamada_kawai(G)\n\n    See Also\n    --------\n    :func:`~networkx.drawing.layout.kamada_kawai_layout`\n    '
    draw(G, kamada_kawai_layout(G), **kwargs)

def draw_random(G, **kwargs):
    if False:
        while True:
            i = 10
    'Draw the graph `G` with a random layout.\n\n    This is a convenience function equivalent to::\n\n        nx.draw(G, pos=nx.random_layout(G), **kwargs)\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    kwargs : optional keywords\n        See `draw_networkx` for a description of optional keywords.\n\n    Notes\n    -----\n    The layout is computed each time this function is called.\n    For repeated drawing it is much more efficient to call\n    `~networkx.drawing.layout.random_layout` directly and reuse the result::\n\n        >>> G = nx.complete_graph(5)\n        >>> pos = nx.random_layout(G)\n        >>> nx.draw(G, pos=pos)  # Draw the original graph\n        >>> # Draw a subgraph, reusing the same node positions\n        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")\n\n    Examples\n    --------\n    >>> G = nx.lollipop_graph(4, 3)\n    >>> nx.draw_random(G)\n\n    See Also\n    --------\n    :func:`~networkx.drawing.layout.random_layout`\n    '
    draw(G, random_layout(G), **kwargs)

def draw_spectral(G, **kwargs):
    if False:
        i = 10
        return i + 15
    'Draw the graph `G` with a spectral 2D layout.\n\n    This is a convenience function equivalent to::\n\n        nx.draw(G, pos=nx.spectral_layout(G), **kwargs)\n\n    For more information about how node positions are determined, see\n    `~networkx.drawing.layout.spectral_layout`.\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    kwargs : optional keywords\n        See `draw_networkx` for a description of optional keywords.\n\n    Notes\n    -----\n    The layout is computed each time this function is called.\n    For repeated drawing it is much more efficient to call\n    `~networkx.drawing.layout.spectral_layout` directly and reuse the result::\n\n        >>> G = nx.complete_graph(5)\n        >>> pos = nx.spectral_layout(G)\n        >>> nx.draw(G, pos=pos)  # Draw the original graph\n        >>> # Draw a subgraph, reusing the same node positions\n        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> nx.draw_spectral(G)\n\n    See Also\n    --------\n    :func:`~networkx.drawing.layout.spectral_layout`\n    '
    draw(G, spectral_layout(G), **kwargs)

def draw_spring(G, **kwargs):
    if False:
        print('Hello World!')
    'Draw the graph `G` with a spring layout.\n\n    This is a convenience function equivalent to::\n\n        nx.draw(G, pos=nx.spring_layout(G), **kwargs)\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    kwargs : optional keywords\n        See `draw_networkx` for a description of optional keywords.\n\n    Notes\n    -----\n    `~networkx.drawing.layout.spring_layout` is also the default layout for\n    `draw`, so this function is equivalent to `draw`.\n\n    The layout is computed each time this function is called.\n    For repeated drawing it is much more efficient to call\n    `~networkx.drawing.layout.spring_layout` directly and reuse the result::\n\n        >>> G = nx.complete_graph(5)\n        >>> pos = nx.spring_layout(G)\n        >>> nx.draw(G, pos=pos)  # Draw the original graph\n        >>> # Draw a subgraph, reusing the same node positions\n        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")\n\n    Examples\n    --------\n    >>> G = nx.path_graph(20)\n    >>> nx.draw_spring(G)\n\n    See Also\n    --------\n    draw\n    :func:`~networkx.drawing.layout.spring_layout`\n    '
    draw(G, spring_layout(G), **kwargs)

def draw_shell(G, nlist=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Draw networkx graph `G` with shell layout.\n\n    This is a convenience function equivalent to::\n\n        nx.draw(G, pos=nx.shell_layout(G, nlist=nlist), **kwargs)\n\n    Parameters\n    ----------\n    G : graph\n        A networkx graph\n\n    nlist : list of list of nodes, optional\n        A list containing lists of nodes representing the shells.\n        Default is `None`, meaning all nodes are in a single shell.\n        See `~networkx.drawing.layout.shell_layout` for details.\n\n    kwargs : optional keywords\n        See `draw_networkx` for a description of optional keywords.\n\n    Notes\n    -----\n    The layout is computed each time this function is called.\n    For repeated drawing it is much more efficient to call\n    `~networkx.drawing.layout.shell_layout` directly and reuse the result::\n\n        >>> G = nx.complete_graph(5)\n        >>> pos = nx.shell_layout(G)\n        >>> nx.draw(G, pos=pos)  # Draw the original graph\n        >>> # Draw a subgraph, reusing the same node positions\n        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> shells = [[0], [1, 2, 3]]\n    >>> nx.draw_shell(G, nlist=shells)\n\n    See Also\n    --------\n    :func:`~networkx.drawing.layout.shell_layout`\n    '
    draw(G, shell_layout(G, nlist=nlist), **kwargs)

def draw_planar(G, **kwargs):
    if False:
        while True:
            i = 10
    'Draw a planar networkx graph `G` with planar layout.\n\n    This is a convenience function equivalent to::\n\n        nx.draw(G, pos=nx.planar_layout(G), **kwargs)\n\n    Parameters\n    ----------\n    G : graph\n        A planar networkx graph\n\n    kwargs : optional keywords\n        See `draw_networkx` for a description of optional keywords.\n\n    Raises\n    ------\n    NetworkXException\n        When `G` is not planar\n\n    Notes\n    -----\n    The layout is computed each time this function is called.\n    For repeated drawing it is much more efficient to call\n    `~networkx.drawing.layout.planar_layout` directly and reuse the result::\n\n        >>> G = nx.path_graph(5)\n        >>> pos = nx.planar_layout(G)\n        >>> nx.draw(G, pos=pos)  # Draw the original graph\n        >>> # Draw a subgraph, reusing the same node positions\n        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.draw_planar(G)\n\n    See Also\n    --------\n    :func:`~networkx.drawing.layout.planar_layout`\n    '
    draw(G, planar_layout(G), **kwargs)

def apply_alpha(colors, alpha, elem_list, cmap=None, vmin=None, vmax=None):
    if False:
        print('Hello World!')
    "Apply an alpha (or list of alphas) to the colors provided.\n\n    Parameters\n    ----------\n\n    colors : color string or array of floats (default='r')\n        Color of element. Can be a single color format string,\n        or a sequence of colors with the same length as nodelist.\n        If numeric values are specified they will be mapped to\n        colors using the cmap and vmin,vmax parameters.  See\n        matplotlib.scatter for more details.\n\n    alpha : float or array of floats\n        Alpha values for elements. This can be a single alpha value, in\n        which case it will be applied to all the elements of color. Otherwise,\n        if it is an array, the elements of alpha will be applied to the colors\n        in order (cycling through alpha multiple times if necessary).\n\n    elem_list : array of networkx objects\n        The list of elements which are being colored. These could be nodes,\n        edges or labels.\n\n    cmap : matplotlib colormap\n        Color map for use if colors is a list of floats corresponding to points\n        on a color mapping.\n\n    vmin, vmax : float\n        Minimum and maximum values for normalizing colors if a colormap is used\n\n    Returns\n    -------\n\n    rgba_colors : numpy ndarray\n        Array containing RGBA format values for each of the node colours.\n\n    "
    from itertools import cycle, islice
    import matplotlib as mpl
    import matplotlib.cm
    import matplotlib.colors
    import numpy as np
    if len(colors) == len(elem_list) and isinstance(colors[0], Number):
        mapper = mpl.cm.ScalarMappable(cmap=cmap)
        mapper.set_clim(vmin, vmax)
        rgba_colors = mapper.to_rgba(colors)
    else:
        try:
            rgba_colors = np.array([mpl.colors.colorConverter.to_rgba(colors)])
        except ValueError:
            rgba_colors = np.array([mpl.colors.colorConverter.to_rgba(color) for color in colors])
    try:
        if len(alpha) > len(rgba_colors) or rgba_colors.size == len(elem_list):
            rgba_colors = np.resize(rgba_colors, (len(elem_list), 4))
            rgba_colors[1:, 0] = rgba_colors[0, 0]
            rgba_colors[1:, 1] = rgba_colors[0, 1]
            rgba_colors[1:, 2] = rgba_colors[0, 2]
        rgba_colors[:, 3] = list(islice(cycle(alpha), len(rgba_colors)))
    except TypeError:
        rgba_colors[:, -1] = alpha
    return rgba_colors
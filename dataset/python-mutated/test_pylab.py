"""Unit tests for matplotlib drawing functions."""
import itertools
import os
import warnings
import pytest
mpl = pytest.importorskip('matplotlib')
np = pytest.importorskip('numpy')
mpl.use('PS')
plt = pytest.importorskip('matplotlib.pyplot')
plt.rcParams['text.usetex'] = False
import networkx as nx
barbell = nx.barbell_graph(4, 6)

def test_draw():
    if False:
        for i in range(10):
            print('nop')
    try:
        functions = [nx.draw_circular, nx.draw_kamada_kawai, nx.draw_planar, nx.draw_random, nx.draw_spectral, nx.draw_spring, nx.draw_shell]
        options = [{'node_color': 'black', 'node_size': 100, 'width': 3}]
        for (function, option) in itertools.product(functions, options):
            function(barbell, **option)
            plt.savefig('test.ps')
    finally:
        try:
            os.unlink('test.ps')
        except OSError:
            pass

def test_draw_shell_nlist():
    if False:
        while True:
            i = 10
    try:
        nlist = [list(range(4)), list(range(4, 10)), list(range(10, 14))]
        nx.draw_shell(barbell, nlist=nlist)
        plt.savefig('test.ps')
    finally:
        try:
            os.unlink('test.ps')
        except OSError:
            pass

def test_edge_colormap():
    if False:
        i = 10
        return i + 15
    colors = range(barbell.number_of_edges())
    nx.draw_spring(barbell, edge_color=colors, width=4, edge_cmap=plt.cm.Blues, with_labels=True)

def test_arrows():
    if False:
        return 10
    nx.draw_spring(barbell.to_directed())

@pytest.mark.parametrize(('edge_color', 'expected'), ((None, 'black'), ('r', 'red'), (['r'], 'red'), ((1.0, 1.0, 0.0), 'yellow'), ([(1.0, 1.0, 0.0)], 'yellow'), ((0, 1, 0, 1), 'lime'), ([(0, 1, 0, 1)], 'lime'), ('#0000ff', 'blue'), (['#0000ff'], 'blue')))
@pytest.mark.parametrize('edgelist', (None, [(0, 1)]))
def test_single_edge_color_undirected(edge_color, expected, edgelist):
    if False:
        while True:
            i = 10
    'Tests ways of specifying all edges have a single color for edges\n    drawn with a LineCollection'
    G = nx.path_graph(3)
    drawn_edges = nx.draw_networkx_edges(G, pos=nx.random_layout(G), edgelist=edgelist, edge_color=edge_color)
    assert mpl.colors.same_color(drawn_edges.get_color(), expected)

@pytest.mark.parametrize(('edge_color', 'expected'), ((None, 'black'), ('r', 'red'), (['r'], 'red'), ((1.0, 1.0, 0.0), 'yellow'), ([(1.0, 1.0, 0.0)], 'yellow'), ((0, 1, 0, 1), 'lime'), ([(0, 1, 0, 1)], 'lime'), ('#0000ff', 'blue'), (['#0000ff'], 'blue')))
@pytest.mark.parametrize('edgelist', (None, [(0, 1)]))
def test_single_edge_color_directed(edge_color, expected, edgelist):
    if False:
        return 10
    'Tests ways of specifying all edges have a single color for edges drawn\n    with FancyArrowPatches'
    G = nx.path_graph(3, create_using=nx.DiGraph)
    drawn_edges = nx.draw_networkx_edges(G, pos=nx.random_layout(G), edgelist=edgelist, edge_color=edge_color)
    for fap in drawn_edges:
        assert mpl.colors.same_color(fap.get_edgecolor(), expected)

def test_edge_color_tuple_interpretation():
    if False:
        print('Hello World!')
    'If edge_color is a sequence with the same length as edgelist, then each\n    value in edge_color is mapped onto each edge via colormap.'
    G = nx.path_graph(6, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    for ec in ((0, 0, 1), (0, 0, 1, 1)):
        drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=ec)
        for fap in drawn_edges:
            assert mpl.colors.same_color(fap.get_edgecolor(), ec)
        drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2)], edge_color=ec)
        for fap in drawn_edges:
            assert mpl.colors.same_color(fap.get_edgecolor(), ec)
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3)], edge_color=(0, 0, 1, 1))
    for fap in drawn_edges:
        assert mpl.colors.same_color(fap.get_edgecolor(), 'blue')
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3), (3, 4)], edge_color=(0, 0, 1))
    for fap in drawn_edges:
        assert mpl.colors.same_color(fap.get_edgecolor(), 'blue')
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3)], edge_color=(0, 0, 1))
    assert mpl.colors.same_color(drawn_edges[0].get_edgecolor(), drawn_edges[1].get_edgecolor())
    for fap in drawn_edges:
        assert not mpl.colors.same_color(fap.get_edgecolor(), 'blue')
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3), (3, 4)], edge_color=(0, 0, 1, 1))
    assert mpl.colors.same_color(drawn_edges[0].get_edgecolor(), drawn_edges[1].get_edgecolor())
    assert mpl.colors.same_color(drawn_edges[2].get_edgecolor(), drawn_edges[3].get_edgecolor())
    for fap in drawn_edges:
        assert not mpl.colors.same_color(fap.get_edgecolor(), 'blue')

def test_fewer_edge_colors_than_num_edges_directed():
    if False:
        i = 10
        return i + 15
    'Test that the edge colors are cycled when there are fewer specified\n    colors than edges.'
    G = barbell.to_directed()
    pos = nx.random_layout(barbell)
    edgecolors = ('r', 'g', 'b')
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=edgecolors)
    for (fap, expected) in zip(drawn_edges, itertools.cycle(edgecolors)):
        assert mpl.colors.same_color(fap.get_edgecolor(), expected)

def test_more_edge_colors_than_num_edges_directed():
    if False:
        return 10
    'Test that extra edge colors are ignored when there are more specified\n    colors than edges.'
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = nx.random_layout(barbell)
    edgecolors = ('r', 'g', 'b', 'c')
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=edgecolors)
    for (fap, expected) in zip(drawn_edges, edgecolors[:-1]):
        assert mpl.colors.same_color(fap.get_edgecolor(), expected)

def test_edge_color_string_with_global_alpha_undirected():
    if False:
        i = 10
        return i + 15
    edge_collection = nx.draw_networkx_edges(barbell, pos=nx.random_layout(barbell), edgelist=[(0, 1), (1, 2)], edge_color='purple', alpha=0.2)
    ec = edge_collection.get_color().squeeze()
    assert len(edge_collection.get_paths()) == 2
    assert mpl.colors.same_color(ec[:-1], 'purple')
    assert ec[-1] == 0.2

def test_edge_color_string_with_global_alpha_directed():
    if False:
        i = 10
        return i + 15
    drawn_edges = nx.draw_networkx_edges(barbell.to_directed(), pos=nx.random_layout(barbell), edgelist=[(0, 1), (1, 2)], edge_color='purple', alpha=0.2)
    assert len(drawn_edges) == 2
    for fap in drawn_edges:
        ec = fap.get_edgecolor()
        assert mpl.colors.same_color(ec[:-1], 'purple')
        assert ec[-1] == 0.2

@pytest.mark.parametrize('graph_type', (nx.Graph, nx.DiGraph))
def test_edge_width_default_value(graph_type):
    if False:
        for i in range(10):
            print('nop')
    'Test the default linewidth for edges drawn either via LineCollection or\n    FancyArrowPatches.'
    G = nx.path_graph(2, create_using=graph_type)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos)
    if isinstance(drawn_edges, list):
        drawn_edges = drawn_edges[0]
    assert drawn_edges.get_linewidth() == 1

@pytest.mark.parametrize(('edgewidth', 'expected'), ((3, 3), ([3], 3)))
def test_edge_width_single_value_undirected(edgewidth, expected):
    if False:
        return 10
    G = nx.path_graph(4)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, width=edgewidth)
    assert len(drawn_edges.get_paths()) == 3
    assert drawn_edges.get_linewidth() == expected

@pytest.mark.parametrize(('edgewidth', 'expected'), ((3, 3), ([3], 3)))
def test_edge_width_single_value_directed(edgewidth, expected):
    if False:
        i = 10
        return i + 15
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, width=edgewidth)
    assert len(drawn_edges) == 3
    for fap in drawn_edges:
        assert fap.get_linewidth() == expected

@pytest.mark.parametrize('edgelist', ([(0, 1), (1, 2), (2, 3)], None, [(0, 1), (1, 2)]))
def test_edge_width_sequence(edgelist):
    if False:
        while True:
            i = 10
    G = barbell.to_directed()
    pos = nx.random_layout(G)
    widths = (0.5, 2.0, 12.0)
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=widths)
    for (fap, expected_width) in zip(drawn_edges, itertools.cycle(widths)):
        assert fap.get_linewidth() == expected_width

def test_edge_color_with_edge_vmin_vmax():
    if False:
        return 10
    'Test that edge_vmin and edge_vmax properly set the dynamic range of the\n    color map when num edges == len(edge_colors).'
    G = nx.path_graph(3, create_using=nx.DiGraph)
    pos = nx.random_layout(G)
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=[0, 1.0])
    orig_colors = [e.get_edgecolor() for e in drawn_edges]
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=[0.2, 0.8], edge_vmin=0.2, edge_vmax=0.8)
    scaled_colors = [e.get_edgecolor() for e in drawn_edges]
    assert mpl.colors.same_color(orig_colors, scaled_colors)

def test_directed_edges_linestyle_default():
    if False:
        while True:
            i = 10
    'Test default linestyle for edges drawn with FancyArrowPatches.'
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos)
    assert len(drawn_edges) == 3
    for fap in drawn_edges:
        assert fap.get_linestyle() == 'solid'

@pytest.mark.parametrize('style', ('dashed', '--', (1, (1, 1))))
def test_directed_edges_linestyle_single_value(style):
    if False:
        return 10
    'Tests support for specifying linestyles with a single value to be applied to\n    all edges in ``draw_networkx_edges`` for FancyArrowPatch outputs\n    (e.g. directed edges).'
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, style=style)
    assert len(drawn_edges) == 3
    for fap in drawn_edges:
        assert fap.get_linestyle() == style

@pytest.mark.parametrize('style_seq', (['dashed'], ['--'], [(1, (1, 1))], ['--', '-', ':'], ['--', '-'], ['--', '-', ':', '-.']))
def test_directed_edges_linestyle_sequence(style_seq):
    if False:
        while True:
            i = 10
    'Tests support for specifying linestyles with sequences in\n    ``draw_networkx_edges`` for FancyArrowPatch outputs (e.g. directed edges).'
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, style=style_seq)
    assert len(drawn_edges) == 3
    for (fap, style) in zip(drawn_edges, itertools.cycle(style_seq)):
        assert fap.get_linestyle() == style

def test_labels_and_colors():
    if False:
        for i in range(10):
            print('nop')
    G = nx.cubical_graph()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=[0, 1, 2, 3], node_color='r', node_size=500, alpha=0.75)
    nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6, 7], node_color='b', node_size=500, alpha=[0.25, 0.5, 0.75, 1.0])
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3), (3, 0)], width=8, alpha=0.5, edge_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)], width=8, alpha=0.5, edge_color='b')
    nx.draw_networkx_edges(G, pos, edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)], arrows=True, min_source_margin=0.5, min_target_margin=0.75, width=8, edge_color='b')
    labels = {}
    labels[0] = '$a$'
    labels[1] = '$b$'
    labels[2] = '$c$'
    labels[3] = '$d$'
    labels[4] = '$\\alpha$'
    labels[5] = '$\\beta$'
    labels[6] = '$\\gamma$'
    labels[7] = '$\\delta$'
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=None, rotate=False)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(4, 5): '4-5'})

@pytest.mark.mpl_image_compare
def test_house_with_colors():
    if False:
        print('Hello World!')
    G = nx.house_graph()
    (fig, ax) = plt.subplots()
    pos = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0.5, 2.0)}
    nx.draw_networkx_nodes(G, pos, node_size=3000, nodelist=[0, 1, 2, 3], node_color='tab:blue')
    nx.draw_networkx_nodes(G, pos, node_size=2000, nodelist=[4], node_color='tab:orange')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=6)
    ax.margins(0.11)
    plt.tight_layout()
    plt.axis('off')
    return fig

def test_axes():
    if False:
        print('Hello World!')
    (fig, ax) = plt.subplots()
    nx.draw(barbell, ax=ax)
    nx.draw_networkx_edge_labels(barbell, nx.circular_layout(barbell), ax=ax)

def test_empty_graph():
    if False:
        for i in range(10):
            print('nop')
    G = nx.Graph()
    nx.draw(G)

def test_draw_empty_nodes_return_values():
    if False:
        print('Hello World!')
    import matplotlib.collections
    G = nx.Graph([(1, 2), (2, 3)])
    DG = nx.DiGraph([(1, 2), (2, 3)])
    pos = nx.circular_layout(G)
    assert isinstance(nx.draw_networkx_nodes(G, pos, nodelist=[]), mpl.collections.PathCollection)
    assert isinstance(nx.draw_networkx_nodes(DG, pos, nodelist=[]), mpl.collections.PathCollection)
    assert nx.draw_networkx_edges(G, pos, edgelist=[], arrows=True) == []
    assert nx.draw_networkx_edges(G, pos, edgelist=[], arrows=False) == []
    assert nx.draw_networkx_edges(DG, pos, edgelist=[], arrows=False) == []
    assert nx.draw_networkx_edges(DG, pos, edgelist=[], arrows=True) == []

def test_multigraph_edgelist_tuples():
    if False:
        while True:
            i = 10
    G = nx.path_graph(3, create_using=nx.MultiDiGraph)
    nx.draw_networkx(G, edgelist=[(0, 1, 0)])
    nx.draw_networkx(G, edgelist=[(0, 1, 0)], node_size=[10, 20, 0])

def test_alpha_iter():
    if False:
        for i in range(10):
            print('nop')
    pos = nx.random_layout(barbell)
    fig = plt.figure()
    fig.add_subplot(131)
    nx.draw_networkx_nodes(barbell, pos, alpha=[0.1, 0.2])
    num_nodes = len(barbell.nodes)
    alpha = [x / num_nodes for x in range(num_nodes)]
    colors = range(num_nodes)
    fig.add_subplot(132)
    nx.draw_networkx_nodes(barbell, pos, node_color=colors, alpha=alpha)
    alpha.append(1)
    fig.add_subplot(133)
    nx.draw_networkx_nodes(barbell, pos, alpha=alpha)

def test_error_invalid_kwds():
    if False:
        return 10
    with pytest.raises(ValueError, match='Received invalid argument'):
        nx.draw(barbell, foo='bar')

def test_draw_networkx_arrowsize_incorrect_size():
    if False:
        i = 10
        return i + 15
    G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 3)])
    arrowsize = [1, 2, 3]
    with pytest.raises(ValueError, match='arrowsize should have the same length as edgelist'):
        nx.draw(G, arrowsize=arrowsize)

@pytest.mark.parametrize('arrowsize', (30, [10, 20, 30]))
def test_draw_edges_arrowsize(arrowsize):
    if False:
        for i in range(10):
            print('nop')
    G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    pos = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
    edges = nx.draw_networkx_edges(G, pos=pos, arrowsize=arrowsize)
    arrowsize = itertools.repeat(arrowsize) if isinstance(arrowsize, int) else arrowsize
    for (fap, expected) in zip(edges, arrowsize):
        assert isinstance(fap, mpl.patches.FancyArrowPatch)
        assert fap.get_mutation_scale() == expected

def test_np_edgelist():
    if False:
        return 10
    nx.draw_networkx(barbell, edgelist=np.array([(0, 2), (0, 3)]))

def test_draw_nodes_missing_node_from_position():
    if False:
        print('Hello World!')
    G = nx.path_graph(3)
    pos = {0: (0, 0), 1: (1, 1)}
    with pytest.raises(nx.NetworkXError, match='has no position'):
        nx.draw_networkx_nodes(G, pos)

@pytest.mark.parametrize('node_shape', ('o', 's'))
def test_draw_edges_min_source_target_margins(node_shape):
    if False:
        return 10
    "Test that there is a wider gap between the node and the start of an\n    incident edge when min_source_margin is specified.\n\n    This test checks that the use of min_{source/target}_margin kwargs result\n    in shorter (more padding) between the edges and source and target nodes.\n    As a crude visual example, let 's' and 't' represent source and target\n    nodes, respectively:\n\n       Default:\n       s-----------------------------t\n\n       With margins:\n       s   -----------------------   t\n\n    "
    (fig, ax) = plt.subplots()
    G = nx.DiGraph([(0, 1)])
    pos = {0: (0, 0), 1: (1, 0)}
    default_patch = nx.draw_networkx_edges(G, pos, ax=ax, node_shape=node_shape)[0]
    default_extent = default_patch.get_extents().corners()[::2, 0]
    padded_patch = nx.draw_networkx_edges(G, pos, ax=ax, node_shape=node_shape, min_source_margin=100, min_target_margin=100)[0]
    padded_extent = padded_patch.get_extents().corners()[::2, 0]
    assert padded_extent[0] > default_extent[0]
    assert padded_extent[1] < default_extent[1]

def test_nonzero_selfloop_with_single_node():
    if False:
        print('Hello World!')
    'Ensure that selfloop extent is non-zero when there is only one node.'
    (fig, ax) = plt.subplots()
    G = nx.DiGraph()
    G.add_node(0)
    G.add_edge(0, 0)
    patch = nx.draw_networkx_edges(G, {0: (0, 0)})[0]
    bbox = patch.get_extents()
    assert bbox.width > 0 and bbox.height > 0
    plt.delaxes(ax)

def test_nonzero_selfloop_with_single_edge_in_edgelist():
    if False:
        i = 10
        return i + 15
    'Ensure that selfloop extent is non-zero when only a single edge is\n    specified in the edgelist.\n    '
    (fig, ax) = plt.subplots()
    G = nx.path_graph(2, create_using=nx.DiGraph)
    G.add_edge(1, 1)
    pos = {n: (n, n) for n in G.nodes}
    patch = nx.draw_networkx_edges(G, pos, edgelist=[(1, 1)])[0]
    bbox = patch.get_extents()
    assert bbox.width > 0 and bbox.height > 0
    plt.delaxes(ax)

def test_apply_alpha():
    if False:
        i = 10
        return i + 15
    'Test apply_alpha when there is a mismatch between the number of\n    supplied colors and elements.\n    '
    nodelist = [0, 1, 2]
    colorlist = ['r', 'g', 'b']
    alpha = 0.5
    rgba_colors = nx.drawing.nx_pylab.apply_alpha(colorlist, alpha, nodelist)
    assert all(rgba_colors[:, -1] == alpha)

def test_draw_edges_toggling_with_arrows_kwarg():
    if False:
        print('Hello World!')
    '\n    The `arrows` keyword argument is used as a 3-way switch to select which\n    type of object to use for drawing edges:\n      - ``arrows=None`` -> default (FancyArrowPatches for directed, else LineCollection)\n      - ``arrows=True`` -> FancyArrowPatches\n      - ``arrows=False`` -> LineCollection\n    '
    import matplotlib.collections
    import matplotlib.patches
    UG = nx.path_graph(3)
    DG = nx.path_graph(3, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in UG}
    for G in (UG, DG):
        edges = nx.draw_networkx_edges(G, pos, arrows=True)
        assert len(edges) == len(G.edges)
        assert isinstance(edges[0], mpl.patches.FancyArrowPatch)
    for G in (UG, DG):
        edges = nx.draw_networkx_edges(G, pos, arrows=False)
        assert isinstance(edges, mpl.collections.LineCollection)
    edges = nx.draw_networkx_edges(UG, pos)
    assert isinstance(edges, mpl.collections.LineCollection)
    edges = nx.draw_networkx_edges(DG, pos)
    assert len(edges) == len(G.edges)
    assert isinstance(edges[0], mpl.patches.FancyArrowPatch)

@pytest.mark.parametrize('drawing_func', (nx.draw, nx.draw_networkx))
def test_draw_networkx_arrows_default_undirected(drawing_func):
    if False:
        i = 10
        return i + 15
    import matplotlib.collections
    G = nx.path_graph(3)
    (fig, ax) = plt.subplots()
    drawing_func(G, ax=ax)
    assert any((isinstance(c, mpl.collections.LineCollection) for c in ax.collections))
    assert not ax.patches
    plt.delaxes(ax)

@pytest.mark.parametrize('drawing_func', (nx.draw, nx.draw_networkx))
def test_draw_networkx_arrows_default_directed(drawing_func):
    if False:
        for i in range(10):
            print('nop')
    import matplotlib.collections
    G = nx.path_graph(3, create_using=nx.DiGraph)
    (fig, ax) = plt.subplots()
    drawing_func(G, ax=ax)
    assert not any((isinstance(c, mpl.collections.LineCollection) for c in ax.collections))
    assert ax.patches
    plt.delaxes(ax)

def test_edgelist_kwarg_not_ignored():
    if False:
        i = 10
        return i + 15
    G = nx.path_graph(3)
    G.add_edge(0, 0)
    (fig, ax) = plt.subplots()
    nx.draw(G, edgelist=[(0, 1), (1, 2)], ax=ax)
    assert not ax.patches
    plt.delaxes(ax)

def test_draw_networkx_edge_label_multiedge_exception():
    if False:
        i = 10
        return i + 15
    '\n    draw_networkx_edge_labels should raise an informative error message when\n    the edge label includes keys\n    '
    exception_msg = 'draw_networkx_edge_labels does not support multiedges'
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(0, 1, weight=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    pos = {n: (n, n) for n in G}
    with pytest.raises(nx.NetworkXError, match=exception_msg):
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

def test_draw_networkx_edge_label_empty_dict():
    if False:
        for i in range(10):
            print('nop')
    'Regression test for draw_networkx_edge_labels with empty dict. See\n    gh-5372.'
    G = nx.path_graph(3)
    pos = {n: (n, n) for n in G.nodes}
    assert nx.draw_networkx_edge_labels(G, pos, edge_labels={}) == {}

def test_draw_networkx_edges_undirected_selfloop_colors():
    if False:
        for i in range(10):
            print('nop')
    'When an edgelist is supplied along with a sequence of colors, check that\n    the self-loops have the correct colors.'
    (fig, ax) = plt.subplots()
    edgelist = [(1, 3), (1, 2), (2, 3), (1, 1), (3, 3), (2, 2)]
    edge_colors = ['pink', 'cyan', 'black', 'red', 'blue', 'green']
    G = nx.Graph(edgelist)
    pos = {n: (n, n) for n in G.nodes}
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edgelist, edge_color=edge_colors)
    assert len(ax.patches) == 3
    sl_points = np.array(edgelist[-3:]) + np.array([0, 0.1])
    for (fap, clr, slp) in zip(ax.patches, edge_colors[-3:], sl_points):
        assert fap.get_path().contains_point(slp)
        assert mpl.colors.same_color(fap.get_edgecolor(), clr)
    plt.delaxes(ax)

@pytest.mark.parametrize('fap_only_kwarg', ({'arrowstyle': '-'}, {'arrowsize': 20}, {'connectionstyle': 'arc3,rad=0.2'}, {'min_source_margin': 10}, {'min_target_margin': 10}))
def test_user_warnings_for_unused_edge_drawing_kwargs(fap_only_kwarg):
    if False:
        print('Hello World!')
    "Users should get a warning when they specify a non-default value for\n    one of the kwargs that applies only to edges drawn with FancyArrowPatches,\n    but FancyArrowPatches aren't being used under the hood."
    G = nx.path_graph(3)
    pos = {n: (n, n) for n in G}
    (fig, ax) = plt.subplots()
    kwarg_name = list(fap_only_kwarg.keys())[0]
    with pytest.warns(UserWarning, match=f'\n\nThe {kwarg_name} keyword argument is not applicable'):
        nx.draw_networkx_edges(G, pos, ax=ax, **fap_only_kwarg)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, **fap_only_kwarg)
    plt.delaxes(ax)
from ray.dag import DAGNode
import os
import tempfile
from ray.dag.utils import _DAGNodeNameGenerator
from ray.util.annotations import DeveloperAPI

@DeveloperAPI
def plot(dag: DAGNode, to_file=None):
    if False:
        return 10
    if to_file is None:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png')
        to_file = tmp_file.name
        extension = 'png'
    else:
        (_, extension) = os.path.splitext(to_file)
        if not extension:
            extension = 'png'
        else:
            extension = extension[1:]
    graph = _dag_to_dot(dag)
    graph.write(to_file, format=extension)
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except ImportError:
        pass
    try:
        tmp_file.close()
    except NameError:
        pass

def _check_pydot_and_graphviz():
    if False:
        i = 10
        return i + 15
    'Check if pydot and graphviz are installed.\n\n    pydot and graphviz are required for plotting. We check this\n    during runtime rather than adding them to Ray dependencies.\n\n    '
    try:
        import pydot
    except ImportError:
        raise ImportError('pydot is required to plot DAG, install it with `pip install pydot`.')
    try:
        pydot.Dot.create(pydot.Dot())
    except (OSError, pydot.InvocationException):
        raise ImportError('graphviz is required to plot DAG, download it from https://graphviz.gitlab.io/download/')

def _get_nodes_and_edges(dag: DAGNode):
    if False:
        while True:
            i = 10
    'Get all unique nodes and edges in the DAG.\n\n    A basic dfs with memorization to get all unique nodes\n    and edges in the DAG.\n    Unique nodes will be used to generate unique names,\n    while edges will be used to construct the graph.\n    '
    edges = []
    nodes = []

    def _dfs(node):
        if False:
            print('Hello World!')
        nodes.append(node)
        for child_node in node._get_all_child_nodes():
            edges.append((child_node, node))
        return node
    dag.apply_recursive(_dfs)
    return (nodes, edges)

def _dag_to_dot(dag: DAGNode):
    if False:
        while True:
            i = 10
    'Create a Dot graph from dag.\n\n    TODO(lchu):\n    1. add more Dot configs in kwargs,\n    e.g. rankdir, alignment, etc.\n    2. add more contents to graph,\n    e.g. args, kwargs and options of each node\n\n    '
    _check_pydot_and_graphviz()
    import pydot
    graph = pydot.Dot(rankdir='LR')
    (nodes, edges) = _get_nodes_and_edges(dag)
    name_generator = _DAGNodeNameGenerator()
    node_names = {}
    for node in nodes:
        node_names[node] = name_generator.get_node_name(node)
    for edge in edges:
        graph.add_edge(pydot.Edge(node_names[edge[0]], node_names[edge[1]]))
    if len(nodes) == 1 and len(edges) == 0:
        graph.add_node(pydot.Node(node_names[nodes[0]]))
    return graph
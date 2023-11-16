from __future__ import annotations
import queue
from typing import TYPE_CHECKING
from .tracker import DummyTracker
from .variables import VariableBase
SIR_GRAPH_CLUSTER_NAME = 'cluster_sir_part'
if TYPE_CHECKING:
    import graphviz

def try_import_graphviz():
    if False:
        i = 10
        return i + 15
    try:
        import graphviz
        return graphviz
    except ImportError:
        return None

def draw_variable(graph: graphviz.Digraph, var: VariableBase):
    if False:
        for i in range(10):
            print('nop')
    '\n    Draw and colour a node in the graph.\n\n    Args:\n        graph (graphviz.Digraph): The graph to draw the variable.\n        var (VariableBase): The variable to draw.\n\n    Returns:\n        None\n    '
    graph.attr('node', shape='oval', style='filled', fillcolor='aliceblue')
    graph.attr('edge', style='solid')
    graph.node(var.id, str(var))
    tracker = var.tracker
    graph.attr('node', shape='rect', style='filled', fillcolor='beige')
    if isinstance(tracker, DummyTracker):
        graph.attr('edge', style='dashed')
        graph.attr('node', shape='rect', style='filled', fillcolor='goldenrod')
    graph.node(tracker.id, str(tracker))
    graph.edge(tracker.id, var.id)
    graph.attr('node', shape='oval', style='filled', fillcolor='cadetblue')
    graph.attr('edge', style='solid')
    for input in tracker.inputs:
        graph.edge(input.id, tracker.id)

def view_tracker(root_variables: list[VariableBase], filename: str, format: str):
    if False:
        return 10
    "\n    Generates a graph visualization starting from the given root variables and save it to the given file.\n\n    Args:\n        root_variables (list[VariableBase]): The root variables to start the visualization from.\n        filename (str): The name of the file used to save the results of the visualisation.\n        format (str): The format (e.g., `pdf`, `png` and 'svg' etc.) of the file to save the visualization to.\n\n    Returns:\n        None\n    "
    graphviz = try_import_graphviz()
    if graphviz is None:
        print('Cannot import graphviz, please install it first.')
        return
    graph = graphviz.Digraph('graph', filename=filename, format=format)
    visited = set()
    var_queue = queue.Queue()
    for var in root_variables:
        var_queue.put(var)
    while not var_queue.empty():
        var = var_queue.get()
        if var.id in visited:
            continue
        visited.add(var.id)
        if isinstance(var.tracker, DummyTracker):
            with graph.subgraph(name=SIR_GRAPH_CLUSTER_NAME) as sir_part:
                sir_part.attr(color='green')
                draw_variable(sir_part, var)
        else:
            draw_variable(graph, var)
        for input in var.tracker.inputs:
            if input not in var_queue.queue:
                var_queue.put(input)
    graph.render(view=False)
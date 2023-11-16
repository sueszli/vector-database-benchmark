"""
Visualization function for a pass manager. Passes are grouped based on their
flow controller, and coloured based on the type of pass.
"""
import os
import inspect
import tempfile
from qiskit.utils import optionals as _optionals
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from .exceptions import VisualizationError
DEFAULT_STYLE = {AnalysisPass: 'red', TransformationPass: 'blue'}

@_optionals.HAS_GRAPHVIZ.require_in_call
@_optionals.HAS_PYDOT.require_in_call
def pass_manager_drawer(pass_manager, filename=None, style=None, raw=False):
    if False:
        return 10
    '\n    Draws the pass manager.\n\n    This function needs `pydot <https://github.com/pydot/pydot>`__, which in turn needs\n    `Graphviz <https://www.graphviz.org/>`__ to be installed.\n\n    Args:\n        pass_manager (PassManager): the pass manager to be drawn\n        filename (str): file path to save image to\n        style (dict or OrderedDict): keys are the pass classes and the values are\n            the colors to make them. An example can be seen in the DEFAULT_STYLE. An ordered\n            dict can be used to ensure a priority coloring when pass falls into multiple\n            categories. Any values not included in the provided dict will be filled in from\n            the default dict\n        raw (Bool) : True if you want to save the raw Dot output not an image. The\n            default is False.\n    Returns:\n        PIL.Image or None: an in-memory representation of the pass manager. Or None if\n        no image was generated or PIL is not installed.\n    Raises:\n        MissingOptionalLibraryError: when nxpd or pydot not installed.\n        VisualizationError: If raw=True and filename=None.\n\n    Example:\n        .. code-block::\n\n             %matplotlib inline\n            from qiskit import QuantumCircuit\n            from qiskit.compiler import transpile\n            from qiskit.transpiler import PassManager\n            from qiskit.visualization import pass_manager_drawer\n            from qiskit.transpiler.passes import Unroller\n\n            circ = QuantumCircuit(3)\n            circ.ccx(0, 1, 2)\n            circ.draw()\n\n            pass_ = Unroller([\'u1\', \'u2\', \'u3\', \'cx\'])\n            pm = PassManager(pass_)\n            new_circ = pm.run(circ)\n            new_circ.draw(output=\'mpl\')\n\n            pass_manager_drawer(pm, "passmanager.jpg")\n    '
    import pydot
    passes = pass_manager.passes()
    if not style:
        style = DEFAULT_STYLE
    graph = pydot.Dot()
    component_id = 0
    prev_node = None
    for (index, controller_group) in enumerate(passes):
        (subgraph, component_id, prev_node) = draw_subgraph(controller_group, component_id, style, prev_node, index)
        graph.add_subgraph(subgraph)
    output = make_output(graph, raw, filename)
    return output

def _get_node_color(pss, style):
    if False:
        print('Hello World!')
    for (typ, color) in style.items():
        if isinstance(pss, typ):
            return color
    for (typ, color) in DEFAULT_STYLE.items():
        if isinstance(pss, typ):
            return color
    return 'black'

@_optionals.HAS_GRAPHVIZ.require_in_call
@_optionals.HAS_PYDOT.require_in_call
def staged_pass_manager_drawer(pass_manager, filename=None, style=None, raw=False):
    if False:
        print('Hello World!')
    '\n    Draws the staged pass manager.\n\n        This function needs `pydot <https://github.com/erocarrera/pydot>`__, which in turn needs\n    `Graphviz <https://www.graphviz.org/>`__ to be installed.\n\n    Args:\n        pass_manager (StagedPassManager): the staged pass manager to be drawn\n        filename (str): file path to save image to\n        style (dict or OrderedDict): keys are the pass classes and the values are\n            the colors to make them. An example can be seen in the DEFAULT_STYLE. An ordered\n            dict can be used to ensure a priority coloring when pass falls into multiple\n            categories. Any values not included in the provided dict will be filled in from\n            the default dict\n        raw (Bool) : True if you want to save the raw Dot output not an image. The\n            default is False.\n    Returns:\n        PIL.Image or None: an in-memory representation of the pass manager. Or None if\n        no image was generated or PIL is not installed.\n    Raises:\n        MissingOptionalLibraryError: when nxpd or pydot not installed.\n        VisualizationError: If raw=True and filename=None.\n\n    Example:\n        .. code-block::\n\n            %matplotlib inline\n            from qiskit.providers.fake_provider import FakeLagosV2\n            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager\n\n            pass_manager = generate_preset_pass_manager(3, FakeLagosV2())\n            pass_manager.draw()\n    '
    import pydot
    stages = list(filter(lambda s: s is not None, pass_manager.expanded_stages))
    if not style:
        style = DEFAULT_STYLE
    graph = pydot.Dot()
    component_id = 0
    idx = 0
    prev_node = None
    for st in stages:
        stage = getattr(pass_manager, st)
        if stage is not None:
            passes = stage.passes()
            stagegraph = pydot.Cluster(str(st), label=str(st), fontname='helvetica', labeljust='l')
            for controller_group in passes:
                (subgraph, component_id, prev_node) = draw_subgraph(controller_group, component_id, style, prev_node, idx)
                stagegraph.add_subgraph(subgraph)
                idx += 1
            graph.add_subgraph(stagegraph)
    output = make_output(graph, raw, filename)
    return output

def draw_subgraph(controller_group, component_id, style, prev_node, idx):
    if False:
        while True:
            i = 10
    'Draw subgraph.'
    import pydot
    label = '[{}] {}'.format(idx, ', '.join(controller_group['flow_controllers']))
    subgraph = pydot.Cluster(str(component_id), label=label, fontname='helvetica', labeljust='l')
    component_id += 1
    for pass_ in controller_group['passes']:
        node = pydot.Node(str(component_id), label=str(type(pass_).__name__), color=_get_node_color(pass_, style), shape='rectangle', fontname='helvetica')
        subgraph.add_node(node)
        component_id += 1
        arg_spec = inspect.getfullargspec(pass_.__init__)
        args = arg_spec[0][1:]
        num_optional = len(arg_spec[3]) if arg_spec[3] else 0
        for (arg_index, arg) in enumerate(args):
            nd_style = 'solid'
            if arg_index >= len(args) - num_optional:
                nd_style = 'dashed'
            input_node = pydot.Node(component_id, label=arg, color='black', shape='ellipse', fontsize=10, style=nd_style, fontname='helvetica')
            subgraph.add_node(input_node)
            component_id += 1
            subgraph.add_edge(pydot.Edge(input_node, node))
        if prev_node:
            subgraph.add_edge(pydot.Edge(prev_node, node))
        prev_node = node
    return (subgraph, component_id, prev_node)

def make_output(graph, raw, filename):
    if False:
        return 10
    'Produce output for pass_manager.'
    if raw:
        if filename:
            graph.write(filename, format='raw')
            return None
        else:
            raise VisualizationError('if format=raw, then a filename is required.')
    if not _optionals.HAS_PIL and filename:
        graph.write_png(filename)
        return None
    _optionals.HAS_PIL.require_now('pass manager drawer')
    with tempfile.TemporaryDirectory() as tmpdirname:
        from PIL import Image
        tmppath = os.path.join(tmpdirname, 'pass_manager.png')
        graph.write_png(tmppath)
        image = Image.open(tmppath)
        os.remove(tmppath)
        if filename:
            image.save(filename, 'PNG')
        return image
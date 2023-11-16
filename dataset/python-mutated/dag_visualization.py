"""
Visualization function for DAG circuit representation.
"""
from rustworkx.visualization import graphviz_draw
from qiskit.dagcircuit.dagnode import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit import Qubit, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.converters import dagdependency_to_circuit
from qiskit.utils import optionals as _optionals
from qiskit.exceptions import InvalidFileError
from .exceptions import VisualizationError

@_optionals.HAS_GRAPHVIZ.require_in_call
def dag_drawer(dag, scale=0.7, filename=None, style='color'):
    if False:
        print('Hello World!')
    "Plot the directed acyclic graph (dag) to represent operation dependencies\n    in a quantum circuit.\n\n    This function calls the :func:`~rustworkx.visualization.graphviz_draw` function from the\n    ``rustworkx`` package to draw the DAG.\n\n    Args:\n        dag (DAGCircuit): The dag to draw.\n        scale (float): scaling factor\n        filename (str): file path to save image to (format inferred from name)\n        style (str): 'plain': B&W graph\n                     'color' (default): color input/output/op nodes\n\n    Returns:\n        PIL.Image: if in Jupyter notebook and not saving to file,\n            otherwise None.\n\n    Raises:\n        VisualizationError: when style is not recognized.\n        InvalidFileError: when filename provided is not valid\n\n    Example:\n        .. plot::\n           :include-source:\n\n            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit\n            from qiskit.dagcircuit import DAGCircuit\n            from qiskit.converters import circuit_to_dag\n            from qiskit.visualization import dag_drawer\n\n            q = QuantumRegister(3, 'q')\n            c = ClassicalRegister(3, 'c')\n            circ = QuantumCircuit(q, c)\n            circ.h(q[0])\n            circ.cx(q[0], q[1])\n            circ.measure(q[0], c[0])\n            circ.rz(0.5, q[1]).c_if(c, 2)\n\n            dag = circuit_to_dag(circ)\n            dag_drawer(dag)\n    "
    type_str = str(type(dag))
    register_bit_labels = {bit: f'{reg.name}[{idx}]' for reg in list(dag.qregs.values()) + list(dag.cregs.values()) for (idx, bit) in enumerate(reg)}
    if 'DAGDependency' in type_str:
        from qiskit.visualization.circuit._utils import get_bit_reg_index
        qubit_indices = {bit: index for (index, bit) in enumerate(dag.qubits)}
        clbit_indices = {bit: index for (index, bit) in enumerate(dag.clbits)}
        graph_attrs = {'dpi': str(100 * scale)}
        dag_dep_circ = dagdependency_to_circuit(dag)

        def node_attr_func(node):
            if False:
                for i in range(10):
                    print('nop')
            if style == 'plain':
                return {}
            if style == 'color':
                n = {}
                args = []
                for (count, arg) in enumerate(node.qargs + node.cargs):
                    if count > 4:
                        args.append('...')
                        break
                    if isinstance(arg, Qubit):
                        f_str = f'q_{qubit_indices[arg]}'
                    elif isinstance(arg, Clbit):
                        f_str = f'c_{clbit_indices[arg]}'
                    else:
                        f_str = f'{arg.index}'
                    arg_str = register_bit_labels.get(arg, f_str)
                    args.append(arg_str)
                n['color'] = 'black'
                n['label'] = str(node.node_id) + ': ' + str(node.name) + ' (' + str(args)[1:-1].replace("'", '') + ')'
                if node.name == 'barrier':
                    n['style'] = 'filled'
                    n['fillcolor'] = 'grey'
                elif getattr(node.op, '_directive', False):
                    n['style'] = 'filled'
                    n['fillcolor'] = 'red'
                elif getattr(node.op, 'condition', None):
                    condition = node.op.condition
                    if isinstance(condition, expr.Expr):
                        cond_txt = ' (cond: [Expr]) ('
                    elif isinstance(condition[0], ClassicalRegister):
                        cond_txt = f' (cond: {condition[0].name}, {int(condition[1])}) ('
                    else:
                        (register, bit_index, reg_index) = get_bit_reg_index(dag_dep_circ, condition[0])
                        if register is not None:
                            cond_txt = f' (cond: {register.name}[{reg_index}], {int(condition[1])}) ('
                        else:
                            cond_txt = f' (cond: {bit_index}, {int(condition[1])}) ('
                    n['style'] = 'filled'
                    n['fillcolor'] = 'green'
                    n['label'] = str(node.node_id) + ': ' + str(node.name) + cond_txt + str(args)[1:-1].replace("'", '') + ')'
                elif node.name != 'measure':
                    n['style'] = 'filled'
                    n['fillcolor'] = 'lightblue'
                return n
            else:
                raise VisualizationError('Unrecognized style %s for the dag_drawer.' % style)
        edge_attr_func = None
    else:
        graph_attrs = {'dpi': str(100 * scale)}

        def node_attr_func(node):
            if False:
                print('Hello World!')
            if style == 'plain':
                return {}
            if style == 'color':
                n = {}
                if isinstance(node, DAGOpNode):
                    n['label'] = node.name
                    n['color'] = 'blue'
                    n['style'] = 'filled'
                    n['fillcolor'] = 'lightblue'
                if isinstance(node, DAGInNode):
                    if isinstance(node.wire, Qubit):
                        label = register_bit_labels.get(node.wire, f'q_{dag.find_bit(node.wire).index}')
                    else:
                        label = register_bit_labels.get(node.wire, f'c_{dag.find_bit(node.wire).index}')
                    n['label'] = label
                    n['color'] = 'black'
                    n['style'] = 'filled'
                    n['fillcolor'] = 'green'
                if isinstance(node, DAGOutNode):
                    if isinstance(node.wire, Qubit):
                        label = register_bit_labels.get(node.wire, f'q[{dag.find_bit(node.wire).index}]')
                    else:
                        label = register_bit_labels.get(node.wire, f'c[{dag.find_bit(node.wire).index}]')
                    n['label'] = label
                    n['color'] = 'black'
                    n['style'] = 'filled'
                    n['fillcolor'] = 'red'
                return n
            else:
                raise VisualizationError('Invalid style %s' % style)

        def edge_attr_func(edge):
            if False:
                print('Hello World!')
            e = {}
            if isinstance(edge, Qubit):
                label = register_bit_labels.get(edge, f'q_{dag.find_bit(edge).index}')
            else:
                label = register_bit_labels.get(edge, f'c_{dag.find_bit(edge).index}')
            e['label'] = label
            return e
    image_type = None
    if filename:
        if '.' not in filename:
            raise InvalidFileError("Parameter 'filename' must be in format 'name.extension'")
        image_type = filename.split('.')[-1]
    return graphviz_draw(dag._multi_graph, node_attr_func, edge_attr_func, graph_attrs, filename, image_type)
"""
A generic InverseCancellation pass for any set of gate-inverse pairs.
"""
from typing import List, Tuple, Union
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

class InverseCancellation(TransformationPass):
    """Cancel specific Gates which are inverses of each other when they occur back-to-
    back."""

    def __init__(self, gates_to_cancel: List[Union[Gate, Tuple[Gate, Gate]]]):
        if False:
            for i in range(10):
                print('nop')
        'Initialize InverseCancellation pass.\n\n        Args:\n            gates_to_cancel: list of gates to cancel\n\n        Raises:\n            TranspilerError:\n                Initialization raises an error when the input is not a self-inverse gate\n                or a two-tuple of inverse gates.\n        '
        for gates in gates_to_cancel:
            if isinstance(gates, Gate):
                if gates != gates.inverse():
                    raise TranspilerError(f'Gate {gates.name} is not self-inverse')
            elif isinstance(gates, tuple):
                if len(gates) != 2:
                    raise TranspilerError(f'Too many or too few inputs: {gates}. Only two are allowed.')
                if gates[0] != gates[1].inverse():
                    raise TranspilerError(f'Gate {gates[0].name} and {gates[1].name} are not inverse.')
            else:
                raise TranspilerError('InverseCancellation pass does not take input type {}. Input must be a Gate.'.format(type(gates)))
        self.self_inverse_gates = []
        self.inverse_gate_pairs = []
        self.self_inverse_gate_names = set()
        self.inverse_gate_pairs_names = set()
        for gates in gates_to_cancel:
            if isinstance(gates, Gate):
                self.self_inverse_gates.append(gates)
                self.self_inverse_gate_names.add(gates.name)
            else:
                self.inverse_gate_pairs.append(gates)
                self.inverse_gate_pairs_names.update((x.name for x in gates))
        super().__init__()

    def run(self, dag: DAGCircuit):
        if False:
            print('Hello World!')
        'Run the InverseCancellation pass on `dag`.\n\n        Args:\n            dag: the directed acyclic graph to run on.\n\n        Returns:\n            DAGCircuit: Transformed DAG.\n        '
        if self.self_inverse_gates:
            dag = self._run_on_self_inverse(dag)
        if self.inverse_gate_pairs:
            dag = self._run_on_inverse_pairs(dag)
        return dag

    def _run_on_self_inverse(self, dag: DAGCircuit):
        if False:
            while True:
                i = 10
        '\n        Run self-inverse gates on `dag`.\n\n        Args:\n            dag: the directed acyclic graph to run on.\n            self_inverse_gates: list of gates who cancel themeselves in pairs\n\n        Returns:\n            DAGCircuit: Transformed DAG.\n        '
        op_counts = dag.count_ops()
        if not self.self_inverse_gate_names.intersection(op_counts):
            return dag
        for gate in self.self_inverse_gates:
            gate_name = gate.name
            gate_count = op_counts.get(gate_name, 0)
            if gate_count <= 1:
                continue
            gate_runs = dag.collect_runs([gate_name])
            for gate_cancel_run in gate_runs:
                partitions = []
                chunk = []
                for i in range(len(gate_cancel_run) - 1):
                    if gate_cancel_run[i].op == gate:
                        chunk.append(gate_cancel_run[i])
                    else:
                        if chunk:
                            partitions.append(chunk)
                            chunk = []
                        continue
                    if gate_cancel_run[i].qargs != gate_cancel_run[i + 1].qargs:
                        partitions.append(chunk)
                        chunk = []
                chunk.append(gate_cancel_run[-1])
                partitions.append(chunk)
                for chunk in partitions:
                    if len(chunk) % 2 == 0:
                        dag.remove_op_node(chunk[0])
                    for node in chunk[1:]:
                        dag.remove_op_node(node)
        return dag

    def _run_on_inverse_pairs(self, dag: DAGCircuit):
        if False:
            print('Hello World!')
        '\n        Run inverse gate pairs on `dag`.\n\n        Args:\n            dag: the directed acyclic graph to run on.\n            inverse_gate_pairs: list of gates with inverse angles that cancel each other.\n\n        Returns:\n            DAGCircuit: Transformed DAG.\n        '
        op_counts = dag.count_ops()
        if not self.inverse_gate_pairs_names.intersection(op_counts):
            return dag
        for pair in self.inverse_gate_pairs:
            gate_0_name = pair[0].name
            gate_1_name = pair[1].name
            if gate_0_name not in op_counts or gate_1_name not in op_counts:
                continue
            gate_cancel_runs = dag.collect_runs([gate_0_name, gate_1_name])
            for dag_nodes in gate_cancel_runs:
                i = 0
                while i < len(dag_nodes) - 1:
                    if dag_nodes[i].qargs == dag_nodes[i + 1].qargs and dag_nodes[i].op == pair[0] and (dag_nodes[i + 1].op == pair[1]):
                        dag.remove_op_node(dag_nodes[i])
                        dag.remove_op_node(dag_nodes[i + 1])
                        i = i + 2
                    elif dag_nodes[i].qargs == dag_nodes[i + 1].qargs and dag_nodes[i].op == pair[1] and (dag_nodes[i + 1].op == pair[0]):
                        dag.remove_op_node(dag_nodes[i])
                        dag.remove_op_node(dag_nodes[i + 1])
                        i = i + 2
                    else:
                        i = i + 1
        return dag
"""
Given a template and a circuit: it applies template matching and substitutes
all compatible maximal matches that reduces the size of the circuit.

**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_
"""
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit import DAGDependency
from qiskit.converters.circuit_to_dagdependency import circuit_to_dagdependency
from qiskit.converters.dagdependency_to_circuit import dagdependency_to_circuit
from qiskit.converters.dag_to_dagdependency import dag_to_dagdependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library.templates import template_nct_2a_1, template_nct_2a_2, template_nct_2a_3
from qiskit.quantum_info.operators.operator import Operator
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.optimization.template_matching import TemplateMatching, TemplateSubstitution, MaximalMatches

class TemplateOptimization(TransformationPass):
    """
    Class for the template optimization pass.
    """

    def __init__(self, template_list=None, heuristics_qubits_param=None, heuristics_backward_param=None, user_cost_dict=None):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            template_list (list[QuantumCircuit()]): list of the different template circuit to apply.\n            heuristics_backward_param (list[int]): [length, survivor] Those are the parameters for\n                applying heuristics on the backward part of the algorithm. This part of the\n                algorithm creates a tree of matching scenario. This tree grows exponentially. The\n                heuristics evaluate which scenarios have the longest match and keep only those.\n                The length is the interval in the tree for cutting it and survivor is the number\n                of scenarios that are kept. We advise to use l=3 and s=1 to have serious time\n                advantage. We remind that the heuristics implies losing a part of the maximal\n                matches. Check reference for more details.\n            heuristics_qubits_param (list[int]): [length] The heuristics for the qubit choice make\n                guesses from the dag dependency of the circuit in order to limit the number of\n                qubit configurations to explore. The length is the number of successors or not\n                predecessors that will be explored in the dag dependency of the circuit, each\n                qubits of the nodes are added to the set of authorized qubits. We advise to use\n                length=1. Check reference for more details.\n            user_cost_dict (Dict[str, int]): quantum cost dictionary passed to TemplateSubstitution\n                to configure its behavior. This will override any default values if None\n                is not given. The key is the name of the gate and the value its quantum cost.\n        '
        super().__init__()
        if template_list is None:
            template_list = [template_nct_2a_1(), template_nct_2a_2(), template_nct_2a_3()]
        self.template_list = template_list
        self.heuristics_qubits_param = heuristics_qubits_param if heuristics_qubits_param is not None else []
        self.heuristics_backward_param = heuristics_backward_param if heuristics_backward_param is not None else []
        self.user_cost_dict = user_cost_dict

    def run(self, dag):
        if False:
            print('Hello World!')
        '\n        Args:\n            dag(DAGCircuit): DAG circuit.\n        Returns:\n            DAGCircuit: optimized DAG circuit.\n        Raises:\n            TranspilerError: If the template has not the right form or\n             if the output circuit acts differently as the input circuit.\n        '
        circuit_dag = dag
        circuit_dag_dep = dag_to_dagdependency(circuit_dag)
        for template in self.template_list:
            if not isinstance(template, (QuantumCircuit, DAGDependency)):
                raise TranspilerError('A template is a Quantumciruit or a DAGDependency.')
            if len(template.qubits) > len(circuit_dag_dep.qubits):
                continue
            identity = np.identity(2 ** len(template.qubits), dtype=complex)
            try:
                if isinstance(template, DAGDependency):
                    data = Operator(dagdependency_to_circuit(template)).data
                else:
                    data = Operator(template).data
                comparison = np.allclose(data, identity)
                if not comparison:
                    raise TranspilerError('A template is a Quantumciruit() that performs the identity.')
            except TypeError:
                pass
            if isinstance(template, QuantumCircuit):
                template_dag_dep = circuit_to_dagdependency(template)
            else:
                template_dag_dep = template
            template_m = TemplateMatching(circuit_dag_dep, template_dag_dep, self.heuristics_qubits_param, self.heuristics_backward_param)
            template_m.run_template_matching()
            matches = template_m.match_list
            if matches:
                maximal = MaximalMatches(matches)
                maximal.run_maximal_matches()
                max_matches = maximal.max_match_list
                substitution = TemplateSubstitution(max_matches, template_m.circuit_dag_dep, template_m.template_dag_dep, self.user_cost_dict)
                substitution.run_dag_opt()
                circuit_dag_dep = substitution.dag_dep_optimized
            else:
                continue
        circuit_dag = dagdependency_to_dag(circuit_dag_dep)
        return circuit_dag
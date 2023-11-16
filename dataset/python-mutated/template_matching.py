"""
Template matching for all possible qubit configurations and initial matches. It
returns the list of all matches obtained from this algorithm.


**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""
import itertools
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.transpiler.passes.optimization.template_matching.forward_match import ForwardMatch
from qiskit.transpiler.passes.optimization.template_matching.backward_match import BackwardMatch

class TemplateMatching:
    """
    Class TemplatingMatching allows to apply the full template matching algorithm.
    """

    def __init__(self, circuit_dag_dep, template_dag_dep, heuristics_qubits_param=None, heuristics_backward_param=None):
        if False:
            return 10
        '\n        Create a TemplateMatching object with necessary arguments.\n        Args:\n            circuit_dag_dep (QuantumCircuit): circuit.\n            template_dag_dep (QuantumCircuit): template.\n            heuristics_backward_param (list[int]): [length, survivor]\n            heuristics_qubits_param (list[int]): [length]\n        '
        self.circuit_dag_dep = circuit_dag_dep
        self.template_dag_dep = template_dag_dep
        self.match_list = []
        self.heuristics_qubits_param = heuristics_qubits_param if heuristics_qubits_param is not None else []
        self.heuristics_backward_param = heuristics_backward_param if heuristics_backward_param is not None else []

    def _list_first_match_new(self, node_circuit, node_template, n_qubits_t, n_clbits_t):
        if False:
            i = 10
            return i + 15
        '\n        Returns the list of qubit for circuit given the first match, the unknown qubit are\n        replaced by -1.\n        Args:\n            node_circuit (DAGDepNode): First match node in the circuit.\n            node_template (DAGDepNode): First match node in the template.\n            n_qubits_t (int): number of qubit in the template.\n            n_clbits_t (int): number of classical bit in the template.\n        Returns:\n            list: list of qubits to consider in circuit (with specific order).\n        '
        l_q = []
        if isinstance(node_circuit.op, ControlledGate) and node_template.op.num_ctrl_qubits > 1:
            control = node_template.op.num_ctrl_qubits
            control_qubits_circuit = node_circuit.qindices[:control]
            not_control_qubits_circuit = node_circuit.qindices[control:]
            if node_template.op.base_gate.name not in ['rxx', 'ryy', 'rzz', 'swap', 'iswap', 'ms']:
                for control_perm_q in itertools.permutations(control_qubits_circuit):
                    control_perm_q = list(control_perm_q)
                    l_q_sub = [-1] * n_qubits_t
                    for q in node_template.qindices:
                        node_circuit_perm = control_perm_q + not_control_qubits_circuit
                        l_q_sub[q] = node_circuit_perm[node_template.qindices.index(q)]
                    l_q.append(l_q_sub)
            else:
                for control_perm_q in itertools.permutations(control_qubits_circuit):
                    control_perm_q = list(control_perm_q)
                    for not_control_perm_q in itertools.permutations(not_control_qubits_circuit):
                        not_control_perm_q = list(not_control_perm_q)
                        l_q_sub = [-1] * n_qubits_t
                        for q in node_template.qindices:
                            node_circuit_perm = control_perm_q + not_control_perm_q
                            l_q_sub[q] = node_circuit_perm[node_template.qindices.index(q)]
                        l_q.append(l_q_sub)
        elif node_template.op.name not in ['rxx', 'ryy', 'rzz', 'swap', 'iswap', 'ms']:
            l_q_sub = [-1] * n_qubits_t
            for q in node_template.qindices:
                l_q_sub[q] = node_circuit.qindices[node_template.qindices.index(q)]
            l_q.append(l_q_sub)
        else:
            for perm_q in itertools.permutations(node_circuit.qindices):
                l_q_sub = [-1] * n_qubits_t
                for q in node_template.qindices:
                    l_q_sub[q] = perm_q[node_template.qindices.index(q)]
                l_q.append(l_q_sub)
        if not node_template.cindices or not node_circuit.cindices:
            l_c = []
        else:
            l_c = [-1] * n_clbits_t
            for c in node_template.cindices:
                l_c[c] = node_circuit[node_template.cindices.index(c)]
        return (l_q, l_c)

    def _sublist(self, lst, exclude, length):
        if False:
            return 10
        '\n        Function that returns all possible combinations of a given length, considering an\n        excluded list of elements.\n        Args:\n            lst (list): list of qubits indices from the circuit.\n            exclude (list): list of qubits from the first matched circuit gate.\n            length (int): length of the list to be returned (number of template qubit -\n            number of qubit from the first matched template gate).\n        Yield:\n            iterator: Iterator of the possible lists.\n        '
        for sublist in itertools.combinations([e for e in lst if e not in exclude], length):
            yield list(sublist)

    def _list_qubit_clbit_circuit(self, list_first_match, permutation):
        if False:
            return 10
        '\n        Function that returns the list of the circuit qubits and clbits give a permutation\n        and an initial match.\n        Args:\n            list_first_match (list): list of qubits indices for the initial match.\n            permutation (list): possible permutation for the circuit qubit.\n        Returns:\n            list: list of circuit qubit for the given permutation and initial match.\n        '
        list_circuit = []
        counter = 0
        for elem in list_first_match:
            if elem == -1:
                list_circuit.append(permutation[counter])
                counter = counter + 1
            else:
                list_circuit.append(elem)
        return list_circuit

    def _add_match(self, backward_match_list):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method to add a match in list only if it is not already in it.\n        If the match is already in the list, the qubit configuration\n        is append to the existing match.\n        Args:\n            backward_match_list (list): match from the backward part of the\n            algorithm.\n        '
        already_in = False
        for b_match in backward_match_list:
            for l_match in self.match_list:
                if b_match.match == l_match.match:
                    index = self.match_list.index(l_match)
                    self.match_list[index].qubit.append(b_match.qubit[0])
                    already_in = True
            if not already_in:
                self.match_list.append(b_match)

    def _explore_circuit(self, node_id_c, node_id_t, n_qubits_t, length):
        if False:
            i = 10
            return i + 15
        "\n        Explore the successors of the node_id_c (up to the given length).\n        Args:\n            node_id_c (int): first match id in the circuit.\n            node_id_t (int): first match id in the template.\n            n_qubits_t (int): number of qubits in the template.\n            length (int): length for exploration of the successors.\n        Returns:\n            list: qubits configuration for the 'length' successors of node_id_c.\n        "
        template_nodes = range(node_id_t + 1, self.template_dag_dep.size())
        circuit_nodes = range(0, self.circuit_dag_dep.size())
        successors_template = self.template_dag_dep.get_node(node_id_t).successors
        counter = 1
        qubit_set = set(self.circuit_dag_dep.get_node(node_id_c).qindices)
        if 2 * len(successors_template) > len(template_nodes):
            successors = self.circuit_dag_dep.get_node(node_id_c).successors
            for succ in successors:
                qarg = self.circuit_dag_dep.get_node(succ).qindices
                if len(qubit_set | set(qarg)) <= n_qubits_t and counter <= length:
                    qubit_set = qubit_set | set(qarg)
                    counter += 1
                elif len(qubit_set | set(qarg)) > n_qubits_t:
                    return list(qubit_set)
            return list(qubit_set)
        else:
            not_successors = list(set(circuit_nodes) - set(self.circuit_dag_dep.get_node(node_id_c).successors))
            candidate = [not_successors[j] for j in range(len(not_successors) - 1, len(not_successors) - 1 - length, -1)]
            for not_succ in candidate:
                qarg = self.circuit_dag_dep.get_node(not_succ).qindices
                if counter <= length and len(qubit_set | set(qarg)) <= n_qubits_t:
                    qubit_set = qubit_set | set(qarg)
                    counter += 1
                elif len(qubit_set | set(qarg)) > n_qubits_t:
                    return list(qubit_set)
            return list(qubit_set)

    def run_template_matching(self):
        if False:
            return 10
        '\n        Run the complete algorithm for finding all maximal matches for the given template and\n        circuit. First it fixes the configuration of the circuit due to the first match.\n        Then it explores all compatible qubit configurations of the circuit. For each\n        qubit configurations, we apply first the Forward part of the algorithm  and then\n        the Backward part of the algorithm. The longest matches for the given configuration\n        are stored. Finally, the list of stored matches is sorted.\n        '
        n_qubits_c = len(self.circuit_dag_dep.qubits)
        n_clbits_c = len(self.circuit_dag_dep.clbits)
        n_qubits_t = len(self.template_dag_dep.qubits)
        n_clbits_t = len(self.template_dag_dep.clbits)
        for template_index in range(0, self.template_dag_dep.size()):
            for circuit_index in range(0, self.circuit_dag_dep.size()):
                if self.circuit_dag_dep.get_node(circuit_index).op.soft_compare(self.template_dag_dep.get_node(template_index).op):
                    qarg_c = self.circuit_dag_dep.get_node(circuit_index).qindices
                    carg_c = self.circuit_dag_dep.get_node(circuit_index).cindices
                    qarg_t = self.template_dag_dep.get_node(template_index).qindices
                    carg_t = self.template_dag_dep.get_node(template_index).cindices
                    node_id_c = circuit_index
                    node_id_t = template_index
                    (all_list_first_match_q, list_first_match_c) = self._list_first_match_new(self.circuit_dag_dep.get_node(circuit_index), self.template_dag_dep.get_node(template_index), n_qubits_t, n_clbits_t)
                    list_circuit_q = list(range(0, n_qubits_c))
                    list_circuit_c = list(range(0, n_clbits_c))
                    if self.heuristics_qubits_param:
                        heuristics_qubits = self._explore_circuit(node_id_c, node_id_t, n_qubits_t, self.heuristics_qubits_param[0])
                    else:
                        heuristics_qubits = []
                    for sub_q in self._sublist(list_circuit_q, qarg_c, n_qubits_t - len(qarg_t)):
                        if set(heuristics_qubits).issubset(set(sub_q) | set(qarg_c)):
                            for perm_q in itertools.permutations(sub_q):
                                perm_q = list(perm_q)
                                for list_first_match_q in all_list_first_match_q:
                                    list_qubit_circuit = self._list_qubit_clbit_circuit(list_first_match_q, perm_q)
                                    if list_circuit_c:
                                        for sub_c in self._sublist(list_circuit_c, carg_c, n_clbits_t - len(carg_t)):
                                            for perm_c in itertools.permutations(sub_c):
                                                perm_c = list(perm_c)
                                                list_clbit_circuit = self._list_qubit_clbit_circuit(list_first_match_c, perm_c)
                                                forward = ForwardMatch(self.circuit_dag_dep, self.template_dag_dep, node_id_c, node_id_t, list_qubit_circuit, list_clbit_circuit)
                                                forward.run_forward_match()
                                                backward = BackwardMatch(forward.circuit_dag_dep, forward.template_dag_dep, forward.match, node_id_c, node_id_t, list_qubit_circuit, list_clbit_circuit, self.heuristics_backward_param)
                                                backward.run_backward_match()
                                                self._add_match(backward.match_final)
                                    else:
                                        forward = ForwardMatch(self.circuit_dag_dep, self.template_dag_dep, node_id_c, node_id_t, list_qubit_circuit)
                                        forward.run_forward_match()
                                        backward = BackwardMatch(forward.circuit_dag_dep, forward.template_dag_dep, forward.match, node_id_c, node_id_t, list_qubit_circuit, [], self.heuristics_backward_param)
                                        backward.run_backward_match()
                                        self._add_match(backward.match_final)
        self.match_list.sort(key=lambda x: len(x.match), reverse=True)
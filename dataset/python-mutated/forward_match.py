"""
Template matching in the forward direction, it takes an initial
match, a configuration of qubit and both circuit and template as inputs. The
result is a list of match between the template and the circuit.


**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""
from qiskit.circuit.controlledgate import ControlledGate

class ForwardMatch:
    """
    Object to apply template matching in the forward direction.
    """

    def __init__(self, circuit_dag_dep, template_dag_dep, node_id_c, node_id_t, qubits, clbits=None):
        if False:
            while True:
                i = 10
        '\n        Create a ForwardMatch class with necessary arguments.\n        Args:\n            circuit_dag_dep (DAGDependency): circuit in the dag dependency form.\n            template_dag_dep (DAGDependency): template in the dag dependency form.\n            node_id_c (int): index of the first gate matched in the circuit.\n            node_id_t (int): index of the first gate matched in the template.\n            qubits (list): list of considered qubits in the circuit.\n            clbits (list): list of considered clbits in the circuit.\n        '
        self.circuit_dag_dep = circuit_dag_dep.copy()
        self.template_dag_dep = template_dag_dep.copy()
        self.qubits = qubits
        self.clbits = clbits if clbits is not None else []
        self.node_id_c = node_id_c
        self.node_id_t = node_id_t
        self.match = []
        self.candidates = []
        self.matched_nodes_list = []
        self.qarg_indices = []
        self.carg_indices = []

    def _init_successors_to_visit(self):
        if False:
            while True:
                i = 10
        "\n        Initialize the attribute list 'SuccessorsToVisit'\n        "
        for i in range(0, self.circuit_dag_dep.size()):
            if i == self.node_id_c:
                self.circuit_dag_dep.get_node(i).successorstovisit = self.circuit_dag_dep.direct_successors(i)

    def _init_matched_with_circuit(self):
        if False:
            print('Hello World!')
        "\n        Initialize the attribute 'MatchedWith' in the template DAG dependency.\n        "
        for i in range(0, self.circuit_dag_dep.size()):
            if i == self.node_id_c:
                self.circuit_dag_dep.get_node(i).matchedwith = [self.node_id_t]
            else:
                self.circuit_dag_dep.get_node(i).matchedwith = []

    def _init_matched_with_template(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Initialize the attribute 'MatchedWith' in the circuit DAG dependency.\n        "
        for i in range(0, self.template_dag_dep.size()):
            if i == self.node_id_t:
                self.template_dag_dep.get_node(i).matchedwith = [self.node_id_c]
            else:
                self.template_dag_dep.get_node(i).matchedwith = []

    def _init_is_blocked_circuit(self):
        if False:
            i = 10
            return i + 15
        "\n        Initialize the attribute 'IsBlocked' in the circuit DAG dependency.\n        "
        for i in range(0, self.circuit_dag_dep.size()):
            self.circuit_dag_dep.get_node(i).isblocked = False

    def _init_is_blocked_template(self):
        if False:
            print('Hello World!')
        "\n        Initialize the attribute 'IsBlocked' in the template DAG dependency.\n        "
        for i in range(0, self.template_dag_dep.size()):
            self.template_dag_dep.get_node(i).isblocked = False

    def _init_list_match(self):
        if False:
            while True:
                i = 10
        '\n        Initialize the list of matched nodes between the circuit and the template\n        with the first match found.\n        '
        self.match.append([self.node_id_t, self.node_id_c])

    def _find_forward_candidates(self, node_id_t):
        if False:
            while True:
                i = 10
        '\n        Find the candidate nodes to be matched in the template for a given node.\n        Args:\n            node_id_t (int): considered node id.\n        '
        matches = []
        for i in range(0, len(self.match)):
            matches.append(self.match[i][0])
        pred = matches.copy()
        if len(pred) > 1:
            pred.sort()
        pred.remove(node_id_t)
        if self.template_dag_dep.direct_successors(node_id_t):
            maximal_index = self.template_dag_dep.direct_successors(node_id_t)[-1]
            for elem in pred:
                if elem > maximal_index:
                    pred.remove(elem)
        block = []
        for node_id in pred:
            for dir_succ in self.template_dag_dep.direct_successors(node_id):
                if dir_succ not in matches:
                    succ = self.template_dag_dep.successors(dir_succ)
                    block = block + succ
        self.candidates = list(set(self.template_dag_dep.direct_successors(node_id_t)) - set(matches) - set(block))

    def _init_matched_nodes(self):
        if False:
            while True:
                i = 10
        '\n        Initialize the list of current matched nodes.\n        '
        self.matched_nodes_list.append([self.node_id_c, self.circuit_dag_dep.get_node(self.node_id_c)])

    def _get_node_forward(self, list_id):
        if False:
            i = 10
            return i + 15
        '\n        Return a node from the matched_node_list for a given list id.\n        Args:\n            list_id (int): considered list id of the desired node.\n\n        Returns:\n            DAGDepNode: DAGDepNode object corresponding to i-th node of the matched_node_list.\n        '
        node = self.matched_nodes_list[list_id][1]
        return node

    def _remove_node_forward(self, list_id):
        if False:
            i = 10
            return i + 15
        '\n        Remove a node of the current matched list for a given list id.\n        Args:\n            list_id (int): considered list id of the desired node.\n        '
        self.matched_nodes_list.pop(list_id)

    def _update_successor(self, node, successor_id):
        if False:
            print('Hello World!')
        "\n        Return a node with an updated attribute 'SuccessorToVisit'.\n        Args:\n            node (DAGDepNode): current node.\n            successor_id (int): successor id to remove.\n\n        Returns:\n            DAGOpNode or DAGOutNode: Node with updated attribute 'SuccessorToVisit'.\n        "
        node_update = node
        node_update.successorstovisit.pop(successor_id)
        return node_update

    def _get_successors_to_visit(self, node, list_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the successor for a given node and id.\n        Args:\n            node (DAGOpNode or DAGOutNode): current node.\n            list_id (int): id in the list for the successor to get.\n\n        Returns:\n            int: id of the successor to get.\n        '
        successor_id = node.successorstovisit[list_id]
        return successor_id

    def _update_qarg_indices(self, qarg):
        if False:
            i = 10
            return i + 15
        '\n        Change qubits indices of the current circuit node in order to\n        be comparable with the indices of the template qubits list.\n        Args:\n            qarg (list): list of qubits indices from the circuit for a given node.\n        '
        self.qarg_indices = []
        for q in qarg:
            if q in self.qubits:
                self.qarg_indices.append(self.qubits.index(q))
        if len(qarg) != len(self.qarg_indices):
            self.qarg_indices = []

    def _update_carg_indices(self, carg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Change clbits indices of the current circuit node in order to\n        be comparable with the indices of the template qubits list.\n        Args:\n            carg (list): list of clbits indices from the circuit for a given node.\n        '
        self.carg_indices = []
        if carg:
            for q in carg:
                if q in self.clbits:
                    self.carg_indices.append(self.clbits.index(q))
            if len(carg) != len(self.carg_indices):
                self.carg_indices = []

    def _is_same_op(self, node_circuit, node_template):
        if False:
            while True:
                i = 10
        '\n        Check if two instructions are the same.\n        Args:\n            node_circuit (DAGDepNode): node in the circuit.\n            node_template (DAGDepNode): node in the template.\n        Returns:\n            bool: True if the same, False otherwise.\n        '
        return node_circuit.op.soft_compare(node_template.op)

    def _is_same_q_conf(self, node_circuit, node_template):
        if False:
            print('Hello World!')
        '\n        Check if the qubits configurations are compatible.\n        Args:\n            node_circuit (DAGDepNode): node in the circuit.\n            node_template (DAGDepNode): node in the template.\n        Returns:\n            bool: True if possible, False otherwise.\n        '
        if isinstance(node_circuit.op, ControlledGate):
            c_template = node_template.op.num_ctrl_qubits
            if c_template == 1:
                return self.qarg_indices == node_template.qindices
            else:
                control_qubits_template = node_template.qindices[:c_template]
                control_qubits_circuit = self.qarg_indices[:c_template]
                if set(control_qubits_circuit) == set(control_qubits_template):
                    target_qubits_template = node_template.qindices[c_template:]
                    target_qubits_circuit = self.qarg_indices[c_template:]
                    if node_template.op.base_gate.name in ['rxx', 'ryy', 'rzz', 'swap', 'iswap', 'ms']:
                        return set(target_qubits_template) == set(target_qubits_circuit)
                    else:
                        return target_qubits_template == target_qubits_circuit
                else:
                    return False
        elif node_template.op.name in ['rxx', 'ryy', 'rzz', 'swap', 'iswap', 'ms']:
            return set(self.qarg_indices) == set(node_template.qindices)
        else:
            return self.qarg_indices == node_template.qindices

    def _is_same_c_conf(self, node_circuit, node_template):
        if False:
            print('Hello World!')
        '\n        Check if the clbits configurations are compatible.\n        Args:\n            node_circuit (DAGDepNode): node in the circuit.\n            node_template (DAGDepNode): node in the template.\n        Returns:\n            bool: True if possible, False otherwise.\n        '
        if node_circuit.type == 'op' and getattr(node_circuit.op, 'condition', None) and (node_template.type == 'op') and getattr(node_template.op, 'condition', None):
            if set(self.carg_indices) != set(node_template.cindices):
                return False
            if getattr(node_circuit.op, 'condition', None)[1] != getattr(node_template.op, 'condition', None)[1]:
                return False
        return True

    def run_forward_match(self):
        if False:
            print('Hello World!')
        '\n        Apply the forward match algorithm and returns the list of matches given an initial match\n        and a circuit qubits configuration.\n        '
        self._init_successors_to_visit()
        self._init_matched_with_circuit()
        self._init_matched_with_template()
        self._init_is_blocked_circuit()
        self._init_is_blocked_template()
        self._init_list_match()
        self._init_matched_nodes()
        while self.matched_nodes_list:
            v_first = self._get_node_forward(0)
            self._remove_node_forward(0)
            if not v_first.successorstovisit:
                continue
            label = self._get_successors_to_visit(v_first, 0)
            v = [label, self.circuit_dag_dep.get_node(label)]
            v_first = self._update_successor(v_first, 0)
            self.matched_nodes_list.append([v_first.node_id, v_first])
            self.matched_nodes_list.sort(key=lambda x: x[1].successorstovisit)
            if v[1].isblocked | (v[1].matchedwith != []):
                continue
            self._find_forward_candidates(v_first.matchedwith[0])
            qarg1 = self.circuit_dag_dep.get_node(label).qindices
            carg1 = self.circuit_dag_dep.get_node(label).cindices
            self._update_qarg_indices(qarg1)
            self._update_carg_indices(carg1)
            match = False
            for i in self.candidates:
                if match:
                    break
                node_circuit = self.circuit_dag_dep.get_node(label)
                node_template = self.template_dag_dep.get_node(i)
                if len(self.qarg_indices) != len(node_template.qindices) or set(self.qarg_indices) != set(node_template.qindices) or node_circuit.name != node_template.name:
                    continue
                if self._is_same_q_conf(node_circuit, node_template) and self._is_same_c_conf(node_circuit, node_template) and self._is_same_op(node_circuit, node_template):
                    v[1].matchedwith = [i]
                    self.template_dag_dep.get_node(i).matchedwith = [label]
                    self.match.append([i, label])
                    potential = self.circuit_dag_dep.direct_successors(label)
                    for potential_id in potential:
                        if self.circuit_dag_dep.get_node(potential_id).isblocked | (self.circuit_dag_dep.get_node(potential_id).matchedwith != []):
                            potential.remove(potential_id)
                    sorted_potential = sorted(potential)
                    v[1].successorstovisit = sorted_potential
                    self.matched_nodes_list.append([v[0], v[1]])
                    self.matched_nodes_list.sort(key=lambda x: x[1].successorstovisit)
                    match = True
                    continue
            if not match:
                v[1].isblocked = True
                for succ in v[1].successors:
                    self.circuit_dag_dep.get_node(succ).isblocked = True
                    if self.circuit_dag_dep.get_node(succ).matchedwith:
                        self.match.remove([self.circuit_dag_dep.get_node(succ).matchedwith[0], succ])
                        match_id = self.circuit_dag_dep.get_node(succ).matchedwith[0]
                        self.template_dag_dep.get_node(match_id).matchedwith = []
                        self.circuit_dag_dep.get_node(succ).matchedwith = []
"""Collect sequences of uninterrupted gates acting on a number of qubits."""
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGOpNode, DAGInNode

class CollectMultiQBlocks(AnalysisPass):
    """Collect sequences of uninterrupted gates acting on groups of qubits.
    ``max_block_size`` specifies the maximum number of qubits that can be acted upon
    by any single group of gates

    Traverse the DAG and find blocks of gates that act consecutively on
    groups of qubits. Write the blocks to ``property_set`` as a list of blocks
    of the form::

        [[g0, g1, g2], [g4, g5]]

    Blocks are reported in a valid topological order. Further, the gates
    within each block are also reported in topological order
    Some gates may not be present in any block (e.g. if the number
    of operands is greater than ``max_block_size``)

    A Disjoint Set Union data structure (DSU) is used to maintain blocks as
    gates are processed. This data structure points each qubit to a set at all
    times and the sets correspond to current blocks. These change over time
    and the data structure allows these changes to be done quickly.
    """

    def __init__(self, max_block_size=2):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.parent = {}
        self.bit_groups = {}
        self.gate_groups = {}
        self.max_block_size = max_block_size

    def find_set(self, index):
        if False:
            while True:
                i = 10
        'DSU function for finding root of set of items\n        If my parent is myself, I am the root. Otherwise we recursively\n        find the root for my parent. After that, we assign my parent to be\n        my root, saving recursion in the future.\n        '
        if index not in self.parent:
            self.parent[index] = index
            self.bit_groups[index] = [index]
            self.gate_groups[index] = []
        if self.parent[index] == index:
            return index
        self.parent[index] = self.find_set(self.parent[index])
        return self.parent[index]

    def union_set(self, set1, set2):
        if False:
            i = 10
            return i + 15
        'DSU function for unioning two sets together\n        Find the roots of each set. Then assign one to have the other\n        as its parent, thus liking the sets.\n        Merges smaller set into larger set in order to have better runtime\n        '
        set1 = self.find_set(set1)
        set2 = self.find_set(set2)
        if set1 == set2:
            return
        if len(self.gate_groups[set1]) < len(self.gate_groups[set2]):
            (set1, set2) = (set2, set1)
        self.parent[set2] = set1
        self.gate_groups[set1].extend(self.gate_groups[set2])
        self.bit_groups[set1].extend(self.bit_groups[set2])
        self.gate_groups[set2].clear()
        self.bit_groups[set2].clear()

    def run(self, dag):
        if False:
            i = 10
            return i + 15
        'Run the CollectMultiQBlocks pass on `dag`.\n\n        The blocks contain "op" nodes in topological sort order\n        such that all gates in a block act on the same set of\n        qubits and are adjacent in the circuit.\n\n        The blocks are built by examining predecessors and successors of\n        "cx" gates in the circuit. u1, u2, u3, cx, id gates will be included.\n\n        After the execution, ``property_set[\'block_list\']`` is set to\n        a list of tuples of ``DAGNode`` objects\n        '
        self.parent = {}
        self.bit_groups = {}
        self.gate_groups = {}
        block_list = []

        def collect_key(x):
            if False:
                for i in range(10):
                    print('nop')
            'special key function for topological ordering.\n            Heuristic for this is to push all gates involving measurement\n            or barriers, etc. as far back as possible (because they force\n            blocks to end). After that, we process gates in order of lowest\n            number of qubits acted on to largest number of qubits acted on\n            because these have less chance of increasing the size of blocks\n            The key also processes all the non operation notes first so that\n            input nodes do not mess with the top sort of op nodes\n            '
            if isinstance(x, DAGInNode):
                return 'a'
            if not isinstance(x, DAGOpNode):
                return 'd'
            if isinstance(x.op, Gate):
                if x.op.is_parameterized() or getattr(x.op, 'condition', None) is not None:
                    return 'c'
                return 'b' + chr(ord('a') + len(x.qargs))
            return 'd'
        op_nodes = dag.topological_op_nodes(key=collect_key)
        for nd in op_nodes:
            can_process = True
            makes_too_big = False
            if getattr(nd.op, 'condition', None) is not None or nd.op.is_parameterized() or (not isinstance(nd.op, Gate)):
                can_process = False
            cur_qubits = {dag.find_bit(bit).index for bit in nd.qargs}
            if can_process:
                c_tops = set()
                for bit in cur_qubits:
                    c_tops.add(self.find_set(bit))
                tot_size = 0
                for group in c_tops:
                    tot_size += len(self.bit_groups[group])
                if tot_size > self.max_block_size:
                    makes_too_big = True
            if not can_process:
                for bit in cur_qubits:
                    bit = self.find_set(bit)
                    if len(self.gate_groups[bit]) == 0:
                        continue
                    block_list.append(self.gate_groups[bit][:])
                    cur_set = set(self.bit_groups[bit])
                    for v in cur_set:
                        self.parent[v] = v
                        self.bit_groups[v] = [v]
                        self.gate_groups[v] = []
            if makes_too_big:
                savings = {}
                tot_size = 0
                for bit in cur_qubits:
                    top = self.find_set(bit)
                    if top in savings:
                        savings[top] = savings[top] - 1
                    else:
                        savings[top] = len(self.bit_groups[top]) - 1
                        tot_size += len(self.bit_groups[top])
                slist = []
                for (item, value) in savings.items():
                    slist.append((value, item))
                slist.sort(reverse=True)
                savings_need = tot_size - self.max_block_size
                for item in slist:
                    if savings_need > 0:
                        savings_need = savings_need - item[0]
                        if len(self.gate_groups[item[1]]) >= 1:
                            block_list.append(self.gate_groups[item[1]][:])
                        cur_set = set(self.bit_groups[item[1]])
                        for v in cur_set:
                            self.parent[v] = v
                            self.bit_groups[v] = [v]
                            self.gate_groups[v] = []
            if can_process:
                if len(cur_qubits) > self.max_block_size:
                    continue
                prev = -1
                for bit in cur_qubits:
                    if prev != -1:
                        self.union_set(prev, bit)
                    prev = bit
                self.gate_groups[self.find_set(prev)].append(nd)
        for index in self.parent:
            if self.parent[index] == index and len(self.gate_groups[index]) != 0:
                block_list.append(self.gate_groups[index][:])
        self.property_set['block_list'] = block_list
        return dag
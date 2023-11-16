"""Object to represent the information at a node in the DAGCircuit."""
from qiskit.exceptions import QiskitError

class DAGDepNode:
    """Object to represent the information at a node in the DAGDependency().

    It is used as the return value from `*_nodes()` functions and can
    be supplied to functions that take a node.
    """
    __slots__ = ['type', '_op', 'name', '_qargs', 'cargs', 'sort_key', 'node_id', 'successors', 'predecessors', 'reachable', 'matchedwith', 'isblocked', 'successorstovisit', 'qindices', 'cindices']

    def __init__(self, type=None, op=None, name=None, qargs=(), cargs=(), successors=None, predecessors=None, reachable=None, matchedwith=None, successorstovisit=None, isblocked=None, qindices=None, cindices=None, nid=-1):
        if False:
            i = 10
            return i + 15
        self.type = type
        self._op = op
        self.name = name
        self._qargs = tuple(qargs) if qargs is not None else ()
        self.cargs = tuple(cargs) if cargs is not None else ()
        self.node_id = nid
        self.sort_key = str(self._qargs)
        self.successors = successors if successors is not None else []
        self.predecessors = predecessors if predecessors is not None else []
        self.reachable = reachable
        self.matchedwith = matchedwith if matchedwith is not None else []
        self.isblocked = isblocked
        self.successorstovisit = successorstovisit if successorstovisit is not None else []
        self.qindices = qindices if qindices is not None else []
        self.cindices = cindices if cindices is not None else []

    @property
    def op(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the Instruction object corresponding to the op for the node, else None'
        if not self.type or self.type != 'op':
            raise QiskitError('The node %s is not an op node' % str(self))
        return self._op

    @op.setter
    def op(self, data):
        if False:
            while True:
                i = 10
        self._op = data

    @property
    def qargs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns list of Qubit, else an empty list.\n        '
        return self._qargs

    @qargs.setter
    def qargs(self, new_qargs):
        if False:
            return 10
        'Sets the qargs to be the given list of qargs.'
        self._qargs = tuple(new_qargs)
        self.sort_key = str(new_qargs)

    @staticmethod
    def semantic_eq(node1, node2):
        if False:
            print('Hello World!')
        '\n        Check if DAG nodes are considered equivalent, e.g., as a node_match for nx.is_isomorphic.\n\n        Args:\n            node1 (DAGDepNode): A node to compare.\n            node2 (DAGDepNode): The other node to compare.\n\n        Return:\n            Bool: If node1 == node2\n        '
        if 'barrier' == node1.name == node2.name:
            return set(node1._qargs) == set(node2._qargs)
        if node1.type == node2.type:
            if node1._op == node2._op:
                if node1.name == node2.name:
                    if node1._qargs == node2._qargs:
                        if node1.cargs == node2.cargs:
                            if node1.type == 'op':
                                if getattr(node1._op, 'condition', None) != getattr(node2._op, 'condition', None):
                                    return False
                            return True
        return False

    def copy(self):
        if False:
            i = 10
            return i + 15
        '\n        Function to copy a DAGDepNode object.\n        Returns:\n            DAGDepNode: a copy of a DAGDepNode object.\n        '
        dagdepnode = DAGDepNode()
        dagdepnode.type = self.type
        dagdepnode._op = self.op
        dagdepnode.name = self.name
        dagdepnode._qargs = self._qargs
        dagdepnode.cargs = self.cargs
        dagdepnode.node_id = self.node_id
        dagdepnode.sort_key = self.sort_key
        dagdepnode.successors = self.successors
        dagdepnode.predecessors = self.predecessors
        dagdepnode.reachable = self.reachable
        dagdepnode.isblocked = self.isblocked
        dagdepnode.successorstovisit = self.successorstovisit
        dagdepnode.qindices = self.qindices
        dagdepnode.cindices = self.cindices
        dagdepnode.matchedwith = self.matchedwith.copy()
        return dagdepnode
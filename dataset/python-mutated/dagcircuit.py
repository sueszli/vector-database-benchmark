"""
Object to represent a quantum circuit as a directed acyclic graph (DAG).

The nodes in the graph are either input/output nodes or operation nodes.
The edges correspond to qubits or bits in the circuit. A directed edge
from node A to node B means that the (qu)bit passes from the output of A
to the input of B. The object's methods allow circuits to be constructed,
composed, and modified. Some natural properties like depth can be computed
directly from the graph.
"""
from collections import OrderedDict, defaultdict, deque, namedtuple
import copy
import math
from typing import Dict, Generator, Any, List
import numpy as np
import rustworkx as rx
from qiskit.circuit import ControlFlowOp, ForLoopOp, IfElseOp, WhileLoopOp, SwitchCaseOp, _classical_resource_map
from qiskit.circuit.controlflow import condition_resources, node_resources, CONTROL_FLOW_OP_NAMES
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode, DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit.bit import Bit
BitLocations = namedtuple('BitLocations', ('index', 'registers'))

class DAGCircuit:
    """
    Quantum circuit as a directed acyclic graph.

    There are 3 types of nodes in the graph: inputs, outputs, and operations.
    The nodes are connected by directed edges that correspond to qubits and
    bits.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Create an empty circuit.'
        self.name = None
        self.metadata = {}
        self._key_cache = {}
        self._wires = set()
        self.input_map = OrderedDict()
        self.output_map = OrderedDict()
        self._multi_graph = rx.PyDAG()
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()
        self.qubits: List[Qubit] = []
        self.clbits: List[Clbit] = []
        self._qubit_indices: Dict[Qubit, BitLocations] = {}
        self._clbit_indices: Dict[Clbit, BitLocations] = {}
        self._global_phase = 0
        self._calibrations = defaultdict(dict)
        self._op_names = {}
        self.duration = None
        self.unit = 'dt'

    @property
    def wires(self):
        if False:
            print('Hello World!')
        'Return a list of the wires in order.'
        return self.qubits + self.clbits

    @property
    def node_counter(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of nodes in the dag.\n        '
        return len(self._multi_graph)

    @property
    def global_phase(self):
        if False:
            while True:
                i = 10
        'Return the global phase of the circuit.'
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle):
        if False:
            while True:
                i = 10
        'Set the global phase of the circuit.\n\n        Args:\n            angle (float, ParameterExpression)\n        '
        if isinstance(angle, ParameterExpression):
            self._global_phase = angle
        else:
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            else:
                self._global_phase = angle % (2 * math.pi)

    @property
    def calibrations(self):
        if False:
            while True:
                i = 10
        "Return calibration dictionary.\n\n        The custom pulse definition of a given gate is of the form\n            {'gate_name': {(qubits, params): schedule}}\n        "
        return dict(self._calibrations)

    @calibrations.setter
    def calibrations(self, calibrations):
        if False:
            for i in range(10):
                print('nop')
        "Set the circuit calibration data from a dictionary of calibration definition.\n\n        Args:\n            calibrations (dict): A dictionary of input in the format\n                {'gate_name': {(qubits, gate_params): schedule}}\n        "
        self._calibrations = defaultdict(dict, calibrations)

    def add_calibration(self, gate, qubits, schedule, params=None):
        if False:
            i = 10
            return i + 15
        'Register a low-level, custom pulse definition for the given gate.\n\n        Args:\n            gate (Union[Gate, str]): Gate information.\n            qubits (Union[int, Tuple[int]]): List of qubits to be measured.\n            schedule (Schedule): Schedule information.\n            params (Optional[List[Union[float, Parameter]]]): A list of parameters.\n\n        Raises:\n            Exception: if the gate is of type string and params is None.\n        '

        def _format(operand):
            if False:
                for i in range(10):
                    print('nop')
            try:
                evaluated = complex(operand)
                if np.isreal(evaluated):
                    evaluated = float(evaluated.real)
                    if evaluated.is_integer():
                        evaluated = int(evaluated)
                return evaluated
            except TypeError:
                return operand
        if isinstance(gate, Gate):
            params = gate.params
            gate = gate.name
        if params is not None:
            params = tuple(map(_format, params))
        else:
            params = ()
        self._calibrations[gate][tuple(qubits), params] = schedule

    def has_calibration_for(self, node):
        if False:
            return 10
        'Return True if the dag has a calibration defined for the node operation. In this\n        case, the operation does not need to be translated to the device basis.\n        '
        if not self.calibrations or node.op.name not in self.calibrations:
            return False
        qubits = tuple((self.qubits.index(qubit) for qubit in node.qargs))
        params = []
        for p in node.op.params:
            if isinstance(p, ParameterExpression) and (not p.parameters):
                params.append(float(p))
            else:
                params.append(p)
        params = tuple(params)
        return (qubits, params) in self.calibrations[node.op.name]

    def remove_all_ops_named(self, opname):
        if False:
            return 10
        'Remove all operation nodes with the given name.'
        for n in self.named_nodes(opname):
            self.remove_op_node(n)

    def add_qubits(self, qubits):
        if False:
            for i in range(10):
                print('nop')
        'Add individual qubit wires.'
        if any((not isinstance(qubit, Qubit) for qubit in qubits)):
            raise DAGCircuitError('not a Qubit instance.')
        duplicate_qubits = set(self.qubits).intersection(qubits)
        if duplicate_qubits:
            raise DAGCircuitError('duplicate qubits %s' % duplicate_qubits)
        for qubit in qubits:
            self.qubits.append(qubit)
            self._qubit_indices[qubit] = BitLocations(len(self.qubits) - 1, [])
            self._add_wire(qubit)

    def add_clbits(self, clbits):
        if False:
            return 10
        'Add individual clbit wires.'
        if any((not isinstance(clbit, Clbit) for clbit in clbits)):
            raise DAGCircuitError('not a Clbit instance.')
        duplicate_clbits = set(self.clbits).intersection(clbits)
        if duplicate_clbits:
            raise DAGCircuitError('duplicate clbits %s' % duplicate_clbits)
        for clbit in clbits:
            self.clbits.append(clbit)
            self._clbit_indices[clbit] = BitLocations(len(self.clbits) - 1, [])
            self._add_wire(clbit)

    def add_qreg(self, qreg):
        if False:
            return 10
        'Add all wires in a quantum register.'
        if not isinstance(qreg, QuantumRegister):
            raise DAGCircuitError('not a QuantumRegister instance.')
        if qreg.name in self.qregs:
            raise DAGCircuitError('duplicate register %s' % qreg.name)
        self.qregs[qreg.name] = qreg
        existing_qubits = set(self.qubits)
        for j in range(qreg.size):
            if qreg[j] in self._qubit_indices:
                self._qubit_indices[qreg[j]].registers.append((qreg, j))
            if qreg[j] not in existing_qubits:
                self.qubits.append(qreg[j])
                self._qubit_indices[qreg[j]] = BitLocations(len(self.qubits) - 1, registers=[(qreg, j)])
                self._add_wire(qreg[j])

    def add_creg(self, creg):
        if False:
            while True:
                i = 10
        'Add all wires in a classical register.'
        if not isinstance(creg, ClassicalRegister):
            raise DAGCircuitError('not a ClassicalRegister instance.')
        if creg.name in self.cregs:
            raise DAGCircuitError('duplicate register %s' % creg.name)
        self.cregs[creg.name] = creg
        existing_clbits = set(self.clbits)
        for j in range(creg.size):
            if creg[j] in self._clbit_indices:
                self._clbit_indices[creg[j]].registers.append((creg, j))
            if creg[j] not in existing_clbits:
                self.clbits.append(creg[j])
                self._clbit_indices[creg[j]] = BitLocations(len(self.clbits) - 1, registers=[(creg, j)])
                self._add_wire(creg[j])

    def _add_wire(self, wire):
        if False:
            return 10
        'Add a qubit or bit to the circuit.\n\n        Args:\n            wire (Bit): the wire to be added\n\n            This adds a pair of in and out nodes connected by an edge.\n\n        Raises:\n            DAGCircuitError: if trying to add duplicate wire\n        '
        if wire not in self._wires:
            self._wires.add(wire)
            inp_node = DAGInNode(wire=wire)
            outp_node = DAGOutNode(wire=wire)
            (input_map_id, output_map_id) = self._multi_graph.add_nodes_from([inp_node, outp_node])
            inp_node._node_id = input_map_id
            outp_node._node_id = output_map_id
            self.input_map[wire] = inp_node
            self.output_map[wire] = outp_node
            self._multi_graph.add_edge(inp_node._node_id, outp_node._node_id, wire)
        else:
            raise DAGCircuitError(f'duplicate wire {wire}')

    def find_bit(self, bit: Bit) -> BitLocations:
        if False:
            while True:
                i = 10
        '\n        Finds locations in the circuit, by mapping the Qubit and Clbit to positional index\n        BitLocations is defined as: BitLocations = namedtuple("BitLocations", ("index", "registers"))\n\n        Args:\n            bit (Bit): The bit to locate.\n\n        Returns:\n            namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)\n                contains the index at which the ``Bit`` can be found (in either\n                :obj:`~DAGCircuit.qubits`, :obj:`~DAGCircuit.clbits`, depending on its\n                type). The second element (``registers``) is a list of ``(register, index)``\n                pairs with an entry for each :obj:`~Register` in the circuit which contains the\n                :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).\n\n          Raises:\n            DAGCircuitError: If the supplied :obj:`~Bit` was of an unknown type.\n            DAGCircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.\n        '
        try:
            if isinstance(bit, Qubit):
                return self._qubit_indices[bit]
            elif isinstance(bit, Clbit):
                return self._clbit_indices[bit]
            else:
                raise DAGCircuitError(f'Could not locate bit of unknown type: {type(bit)}')
        except KeyError as err:
            raise DAGCircuitError(f'Could not locate provided bit: {bit}. Has it been added to the DAGCircuit?') from err

    def remove_clbits(self, *clbits):
        if False:
            i = 10
            return i + 15
        '\n        Remove classical bits from the circuit. All bits MUST be idle.\n        Any registers with references to at least one of the specified bits will\n        also be removed.\n\n        Args:\n            clbits (List[Clbit]): The bits to remove.\n\n        Raises:\n            DAGCircuitError: a clbit is not a :obj:`.Clbit`, is not in the circuit,\n                or is not idle.\n        '
        if any((not isinstance(clbit, Clbit) for clbit in clbits)):
            raise DAGCircuitError('clbits not of type Clbit: %s' % [b for b in clbits if not isinstance(b, Clbit)])
        clbits = set(clbits)
        unknown_clbits = clbits.difference(self.clbits)
        if unknown_clbits:
            raise DAGCircuitError('clbits not in circuit: %s' % unknown_clbits)
        busy_clbits = {bit for bit in clbits if not self._is_wire_idle(bit)}
        if busy_clbits:
            raise DAGCircuitError('clbits not idle: %s' % busy_clbits)
        cregs_to_remove = {creg for creg in self.cregs.values() if not clbits.isdisjoint(creg)}
        self.remove_cregs(*cregs_to_remove)
        for clbit in clbits:
            self._remove_idle_wire(clbit)
            self.clbits.remove(clbit)
            del self._clbit_indices[clbit]
        for (i, clbit) in enumerate(self.clbits):
            self._clbit_indices[clbit] = self._clbit_indices[clbit]._replace(index=i)

    def remove_cregs(self, *cregs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove classical registers from the circuit, leaving underlying bits\n        in place.\n\n        Raises:\n            DAGCircuitError: a creg is not a ClassicalRegister, or is not in\n            the circuit.\n        '
        if any((not isinstance(creg, ClassicalRegister) for creg in cregs)):
            raise DAGCircuitError('cregs not of type ClassicalRegister: %s' % [r for r in cregs if not isinstance(r, ClassicalRegister)])
        unknown_cregs = set(cregs).difference(self.cregs.values())
        if unknown_cregs:
            raise DAGCircuitError('cregs not in circuit: %s' % unknown_cregs)
        for creg in cregs:
            del self.cregs[creg.name]
            for j in range(creg.size):
                bit = creg[j]
                bit_position = self._clbit_indices[bit]
                bit_position.registers.remove((creg, j))

    def remove_qubits(self, *qubits):
        if False:
            i = 10
            return i + 15
        '\n        Remove quantum bits from the circuit. All bits MUST be idle.\n        Any registers with references to at least one of the specified bits will\n        also be removed.\n\n        Args:\n            qubits (List[~qiskit.circuit.Qubit]): The bits to remove.\n\n        Raises:\n            DAGCircuitError: a qubit is not a :obj:`~.circuit.Qubit`, is not in the circuit,\n                or is not idle.\n        '
        if any((not isinstance(qubit, Qubit) for qubit in qubits)):
            raise DAGCircuitError('qubits not of type Qubit: %s' % [b for b in qubits if not isinstance(b, Qubit)])
        qubits = set(qubits)
        unknown_qubits = qubits.difference(self.qubits)
        if unknown_qubits:
            raise DAGCircuitError('qubits not in circuit: %s' % unknown_qubits)
        busy_qubits = {bit for bit in qubits if not self._is_wire_idle(bit)}
        if busy_qubits:
            raise DAGCircuitError('qubits not idle: %s' % busy_qubits)
        qregs_to_remove = {qreg for qreg in self.qregs.values() if not qubits.isdisjoint(qreg)}
        self.remove_qregs(*qregs_to_remove)
        for qubit in qubits:
            self._remove_idle_wire(qubit)
            self.qubits.remove(qubit)
            del self._qubit_indices[qubit]
        for (i, qubit) in enumerate(self.qubits):
            self._qubit_indices[qubit] = self._qubit_indices[qubit]._replace(index=i)

    def remove_qregs(self, *qregs):
        if False:
            i = 10
            return i + 15
        '\n        Remove classical registers from the circuit, leaving underlying bits\n        in place.\n\n        Raises:\n            DAGCircuitError: a qreg is not a QuantumRegister, or is not in\n            the circuit.\n        '
        if any((not isinstance(qreg, QuantumRegister) for qreg in qregs)):
            raise DAGCircuitError('qregs not of type QuantumRegister: %s' % [r for r in qregs if not isinstance(r, QuantumRegister)])
        unknown_qregs = set(qregs).difference(self.qregs.values())
        if unknown_qregs:
            raise DAGCircuitError('qregs not in circuit: %s' % unknown_qregs)
        for qreg in qregs:
            del self.qregs[qreg.name]
            for j in range(qreg.size):
                bit = qreg[j]
                bit_position = self._qubit_indices[bit]
                bit_position.registers.remove((qreg, j))

    def _is_wire_idle(self, wire):
        if False:
            i = 10
            return i + 15
        'Check if a wire is idle.\n\n        Args:\n            wire (Bit): a wire in the circuit.\n\n        Returns:\n            bool: true if the wire is idle, false otherwise.\n\n        Raises:\n            DAGCircuitError: the wire is not in the circuit.\n        '
        if wire not in self._wires:
            raise DAGCircuitError('wire %s not in circuit' % wire)
        try:
            child = next(self.successors(self.input_map[wire]))
        except StopIteration as e:
            raise DAGCircuitError('Invalid dagcircuit input node %s has no output' % self.input_map[wire]) from e
        return child is self.output_map[wire]

    def _remove_idle_wire(self, wire):
        if False:
            print('Hello World!')
        'Remove an idle qubit or bit from the circuit.\n\n        Args:\n            wire (Bit): the wire to be removed, which MUST be idle.\n        '
        inp_node = self.input_map[wire]
        oup_node = self.output_map[wire]
        self._multi_graph.remove_node(inp_node._node_id)
        self._multi_graph.remove_node(oup_node._node_id)
        self._wires.remove(wire)
        del self.input_map[wire]
        del self.output_map[wire]

    def _check_condition(self, name, condition):
        if False:
            while True:
                i = 10
        'Verify that the condition is valid.\n\n        Args:\n            name (string): used for error reporting\n            condition (tuple or None): a condition tuple (ClassicalRegister, int) or (Clbit, bool)\n\n        Raises:\n            DAGCircuitError: if conditioning on an invalid register\n        '
        if condition is None:
            return
        resources = condition_resources(condition)
        for creg in resources.cregs:
            if creg.name not in self.cregs:
                raise DAGCircuitError(f'invalid creg in condition for {name}')
        if not set(resources.clbits).issubset(self.clbits):
            raise DAGCircuitError(f'invalid clbits in condition for {name}')

    def _check_bits(self, args, amap):
        if False:
            print('Hello World!')
        'Check the values of a list of (qu)bit arguments.\n\n        For each element of args, check that amap contains it.\n\n        Args:\n            args (list[Bit]): the elements to be checked\n            amap (dict): a dictionary keyed on Qubits/Clbits\n\n        Raises:\n            DAGCircuitError: if a qubit is not contained in amap\n        '
        for wire in args:
            if wire not in amap:
                raise DAGCircuitError(f'(qu)bit {wire} not found in {amap}')

    @staticmethod
    def _bits_in_operation(operation):
        if False:
            i = 10
            return i + 15
        'Return an iterable over the classical bits that are inherent to an instruction.  This\n        includes a `condition`, or the `target` of a :class:`.ControlFlowOp`.\n\n        Args:\n            instruction: the :class:`~.circuit.Instruction` instance for a node.\n\n        Returns:\n            Iterable[Clbit]: the :class:`.Clbit`\\ s involved.\n        '
        if (condition := getattr(operation, 'condition', None)) is not None:
            yield from condition_resources(condition).clbits
        if isinstance(operation, SwitchCaseOp):
            target = operation.target
            if isinstance(target, Clbit):
                yield target
            elif isinstance(target, ClassicalRegister):
                yield from target
            else:
                yield from node_resources(target).clbits

    @staticmethod
    def _operation_may_have_bits(operation) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return whether a given :class:`.Operation` may contain any :class:`.Clbit` instances\n        in itself (e.g. a control-flow operation).\n\n        Args:\n            operation (qiskit.circuit.Operation): the operation to check.\n        '
        return getattr(operation, 'condition', None) is not None or isinstance(operation, SwitchCaseOp)

    def _increment_op(self, op):
        if False:
            print('Hello World!')
        if op.name in self._op_names:
            self._op_names[op.name] += 1
        else:
            self._op_names[op.name] = 1

    def _decrement_op(self, op):
        if False:
            i = 10
            return i + 15
        if self._op_names[op.name] == 1:
            del self._op_names[op.name]
        else:
            self._op_names[op.name] -= 1

    def copy_empty_like(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of self with the same structure but empty.\n\n        That structure includes:\n            * name and other metadata\n            * global phase\n            * duration\n            * all the qubits and clbits, including the registers.\n\n        Returns:\n            DAGCircuit: An empty copy of self.\n        '
        target_dag = DAGCircuit()
        target_dag.name = self.name
        target_dag._global_phase = self._global_phase
        target_dag.duration = self.duration
        target_dag.unit = self.unit
        target_dag.metadata = self.metadata
        target_dag._key_cache = self._key_cache
        target_dag.add_qubits(self.qubits)
        target_dag.add_clbits(self.clbits)
        for qreg in self.qregs.values():
            target_dag.add_qreg(qreg)
        for creg in self.cregs.values():
            target_dag.add_creg(creg)
        return target_dag

    def apply_operation_back(self, op, qargs=(), cargs=(), *, check=True):
        if False:
            print('Hello World!')
        'Apply an operation to the output of the circuit.\n\n        Args:\n            op (qiskit.circuit.Operation): the operation associated with the DAG node\n            qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to\n            cargs (tuple[Clbit]): cbits that op will be applied to\n            check (bool): If ``True`` (default), this function will enforce that the\n                :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are\n                :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*\n                uphold these invariants itself, but the cost of several checks will be skipped.\n                This is most useful when building a new DAG from a source of known-good nodes.\n        Returns:\n            DAGOpNode: the node for the op that was added to the dag\n\n        Raises:\n            DAGCircuitError: if a leaf node is connected to multiple outputs\n\n        '
        qargs = tuple(qargs)
        cargs = tuple(cargs)
        if self._operation_may_have_bits(op):
            all_cbits = set(self._bits_in_operation(op)).union(cargs)
        else:
            all_cbits = cargs
        if check:
            self._check_condition(op.name, getattr(op, 'condition', None))
            self._check_bits(qargs, self.output_map)
            self._check_bits(all_cbits, self.output_map)
        node = DAGOpNode(op=op, qargs=qargs, cargs=cargs, dag=self)
        node._node_id = self._multi_graph.add_node(node)
        self._increment_op(op)
        self._multi_graph.insert_node_on_in_edges_multiple(node._node_id, [self.output_map[bit]._node_id for bits in (qargs, all_cbits) for bit in bits])
        return node

    def apply_operation_front(self, op, qargs=(), cargs=(), *, check=True):
        if False:
            while True:
                i = 10
        'Apply an operation to the input of the circuit.\n\n        Args:\n            op (qiskit.circuit.Operation): the operation associated with the DAG node\n            qargs (tuple[~qiskit.circuit.Qubit]): qubits that op will be applied to\n            cargs (tuple[Clbit]): cbits that op will be applied to\n            check (bool): If ``True`` (default), this function will enforce that the\n                :class:`.DAGCircuit` data-structure invariants are maintained (all ``qargs`` are\n                :class:`~.circuit.Qubit`\\ s, all are in the DAG, etc).  If ``False``, the caller *must*\n                uphold these invariants itself, but the cost of several checks will be skipped.\n                This is most useful when building a new DAG from a source of known-good nodes.\n        Returns:\n            DAGOpNode: the node for the op that was added to the dag\n\n        Raises:\n            DAGCircuitError: if initial nodes connected to multiple out edges\n        '
        qargs = tuple(qargs)
        cargs = tuple(cargs)
        if self._operation_may_have_bits(op):
            all_cbits = set(self._bits_in_operation(op)).union(cargs)
        else:
            all_cbits = cargs
        if check:
            self._check_condition(op.name, getattr(op, 'condition', None))
            self._check_bits(qargs, self.input_map)
            self._check_bits(all_cbits, self.input_map)
        node = DAGOpNode(op=op, qargs=qargs, cargs=cargs, dag=self)
        node._node_id = self._multi_graph.add_node(node)
        self._increment_op(op)
        self._multi_graph.insert_node_on_out_edges_multiple(node._node_id, [self.input_map[bit]._node_id for bits in (qargs, all_cbits) for bit in bits])
        return node

    def compose(self, other, qubits=None, clbits=None, front=False, inplace=True):
        if False:
            i = 10
            return i + 15
        'Compose the ``other`` circuit onto the output of this circuit.\n\n        A subset of input wires of ``other`` are mapped\n        to a subset of output wires of this circuit.\n\n        ``other`` can be narrower or of equal width to ``self``.\n\n        Args:\n            other (DAGCircuit): circuit to compose with self\n            qubits (list[~qiskit.circuit.Qubit|int]): qubits of self to compose onto.\n            clbits (list[Clbit|int]): clbits of self to compose onto.\n            front (bool): If True, front composition will be performed (not implemented yet)\n            inplace (bool): If True, modify the object. Otherwise return composed circuit.\n\n        Returns:\n            DAGCircuit: the composed dag (returns None if inplace==True).\n\n        Raises:\n            DAGCircuitError: if ``other`` is wider or there are duplicate edge mappings.\n        '
        if front:
            raise DAGCircuitError('Front composition not supported yet.')
        if len(other.qubits) > len(self.qubits) or len(other.clbits) > len(self.clbits):
            raise DAGCircuitError("Trying to compose with another DAGCircuit which has more 'in' edges.")
        identity_qubit_map = dict(zip(other.qubits, self.qubits))
        identity_clbit_map = dict(zip(other.clbits, self.clbits))
        if qubits is None:
            qubit_map = identity_qubit_map
        elif len(qubits) != len(other.qubits):
            raise DAGCircuitError('Number of items in qubits parameter does not match number of qubits in the circuit.')
        else:
            qubit_map = {other.qubits[i]: self.qubits[q] if isinstance(q, int) else q for (i, q) in enumerate(qubits)}
        if clbits is None:
            clbit_map = identity_clbit_map
        elif len(clbits) != len(other.clbits):
            raise DAGCircuitError('Number of items in clbits parameter does not match number of clbits in the circuit.')
        else:
            clbit_map = {other.clbits[i]: self.clbits[c] if isinstance(c, int) else c for (i, c) in enumerate(clbits)}
        edge_map = {**qubit_map, **clbit_map} or None
        if edge_map is None:
            edge_map = {**identity_qubit_map, **identity_clbit_map}
        if len(set(edge_map.values())) != len(edge_map):
            raise DAGCircuitError('duplicates in wire_map')
        if inplace:
            dag = self
        else:
            dag = copy.deepcopy(self)
        dag.global_phase += other.global_phase
        for (gate, cals) in other.calibrations.items():
            dag._calibrations[gate].update(cals)

        def _reject_new_register(reg):
            if False:
                i = 10
                return i + 15
            raise DAGCircuitError(f"No register with '{reg.bits}' to map this expression onto.")
        variable_mapper = _classical_resource_map.VariableMapper(dag.cregs.values(), edge_map, _reject_new_register)
        for nd in other.topological_nodes():
            if isinstance(nd, DAGInNode):
                m_wire = edge_map.get(nd.wire, nd.wire)
                if m_wire not in dag.output_map:
                    raise DAGCircuitError('wire %s[%d] not in self' % (m_wire.register.name, m_wire.index))
                if nd.wire not in other._wires:
                    raise DAGCircuitError('inconsistent wire type for %s[%d] in other' % (nd.register.name, nd.wire.index))
            elif isinstance(nd, DAGOutNode):
                pass
            elif isinstance(nd, DAGOpNode):
                m_qargs = [edge_map.get(x, x) for x in nd.qargs]
                m_cargs = [edge_map.get(x, x) for x in nd.cargs]
                op = nd.op.copy()
                if (condition := getattr(op, 'condition', None)) is not None:
                    if not isinstance(op, ControlFlowOp):
                        op = op.c_if(*variable_mapper.map_condition(condition, allow_reorder=True))
                    else:
                        op.condition = variable_mapper.map_condition(condition, allow_reorder=True)
                elif isinstance(op, SwitchCaseOp):
                    op.target = variable_mapper.map_target(op.target)
                dag.apply_operation_back(op, m_qargs, m_cargs, check=False)
            else:
                raise DAGCircuitError('bad node type %s' % type(nd))
        if not inplace:
            return dag
        else:
            return None

    def reverse_ops(self):
        if False:
            i = 10
            return i + 15
        'Reverse the operations in the ``self`` circuit.\n\n        Returns:\n            DAGCircuit: the reversed dag.\n        '
        from qiskit.converters import dag_to_circuit, circuit_to_dag
        qc = dag_to_circuit(self)
        reversed_qc = qc.reverse_ops()
        reversed_dag = circuit_to_dag(reversed_qc)
        return reversed_dag

    def idle_wires(self, ignore=None):
        if False:
            i = 10
            return i + 15
        'Return idle wires.\n\n        Args:\n            ignore (list(str)): List of node names to ignore. Default: []\n\n        Yields:\n            Bit: Bit in idle wire.\n\n        Raises:\n            DAGCircuitError: If the DAG is invalid\n        '
        if ignore is None:
            ignore = set()
        ignore_set = set(ignore)
        for wire in self._wires:
            if not ignore:
                if self._is_wire_idle(wire):
                    yield wire
            else:
                for node in self.nodes_on_wire(wire, only_ops=True):
                    if node.op.name not in ignore_set:
                        break
                else:
                    yield wire

    def size(self, *, recurse: bool=False):
        if False:
            return 10
        'Return the number of operations.  If there is control flow present, this count may only\n        be an estimate, as the complete control-flow path cannot be statically known.\n\n        Args:\n            recurse: if ``True``, then recurse into control-flow operations.  For loops with\n                known-length iterators are counted unrolled.  If-else blocks sum both of the two\n                branches.  While loops are counted as if the loop body runs once only.  Defaults to\n                ``False`` and raises :class:`.DAGCircuitError` if any control flow is present, to\n                avoid silently returning a mostly meaningless number.\n\n        Returns:\n            int: the circuit size\n\n        Raises:\n            DAGCircuitError: if an unknown :class:`.ControlFlowOp` is present in a call with\n                ``recurse=True``, or any control flow is present in a non-recursive call.\n        '
        length = len(self._multi_graph) - 2 * len(self._wires)
        if not recurse:
            if any((x in self._op_names for x in CONTROL_FLOW_OP_NAMES)):
                raise DAGCircuitError("Size with control flow is ambiguous. You may use `recurse=True` to get a result, but see this method's documentation for the meaning of this.")
            return length
        from qiskit.converters import circuit_to_dag
        for node in self.op_nodes(ControlFlowOp):
            if isinstance(node.op, ForLoopOp):
                indexset = node.op.params[0]
                inner = len(indexset) * circuit_to_dag(node.op.blocks[0]).size(recurse=True)
            elif isinstance(node.op, WhileLoopOp):
                inner = circuit_to_dag(node.op.blocks[0]).size(recurse=True)
            elif isinstance(node.op, (IfElseOp, SwitchCaseOp)):
                inner = sum((circuit_to_dag(block).size(recurse=True) for block in node.op.blocks))
            else:
                raise DAGCircuitError(f"unknown control-flow type: '{node.op.name}'")
            length += inner - 1
        return length

    def depth(self, *, recurse: bool=False):
        if False:
            i = 10
            return i + 15
        'Return the circuit depth.  If there is control flow present, this count may only be an\n        estimate, as the complete control-flow path cannot be statically known.\n\n        Args:\n            recurse: if ``True``, then recurse into control-flow operations.  For loops\n                with known-length iterators are counted as if the loop had been manually unrolled\n                (*i.e.* with each iteration of the loop body written out explicitly).\n                If-else blocks take the longer case of the two branches.  While loops are counted as\n                if the loop body runs once only.  Defaults to ``False`` and raises\n                :class:`.DAGCircuitError` if any control flow is present, to avoid silently\n                returning a nonsensical number.\n\n        Returns:\n            int: the circuit depth\n\n        Raises:\n            DAGCircuitError: if not a directed acyclic graph\n            DAGCircuitError: if unknown control flow is present in a recursive call, or any control\n                flow is present in a non-recursive call.\n        '
        if recurse:
            from qiskit.converters import circuit_to_dag
            node_lookup = {}
            for node in self.op_nodes(ControlFlowOp):
                weight = len(node.op.params[0]) if isinstance(node.op, ForLoopOp) else 1
                if weight == 0:
                    node_lookup[node._node_id] = 0
                else:
                    node_lookup[node._node_id] = weight * max((circuit_to_dag(block).depth(recurse=True) for block in node.op.blocks))

            def weight_fn(_source, target, _edge):
                if False:
                    i = 10
                    return i + 15
                return node_lookup.get(target, 1)
        else:
            if any((x in self._op_names for x in CONTROL_FLOW_OP_NAMES)):
                raise DAGCircuitError("Depth with control flow is ambiguous. You may use `recurse=True` to get a result, but see this method's documentation for the meaning of this.")
            weight_fn = None
        try:
            depth = rx.dag_longest_path_length(self._multi_graph, weight_fn) - 1
        except rx.DAGHasCycle as ex:
            raise DAGCircuitError('not a DAG') from ex
        return depth if depth >= 0 else 0

    def width(self):
        if False:
            print('Hello World!')
        'Return the total number of qubits + clbits used by the circuit.\n        This function formerly returned the number of qubits by the calculation\n        return len(self._wires) - self.num_clbits()\n        but was changed by issue #2564 to return number of qubits + clbits\n        with the new function DAGCircuit.num_qubits replacing the former\n        semantic of DAGCircuit.width().\n        '
        return len(self._wires)

    def num_qubits(self):
        if False:
            i = 10
            return i + 15
        'Return the total number of qubits used by the circuit.\n        num_qubits() replaces former use of width().\n        DAGCircuit.width() now returns qubits + clbits for\n        consistency with Circuit.width() [qiskit-terra #2564].\n        '
        return len(self.qubits)

    def num_clbits(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the total number of classical bits used by the circuit.'
        return len(self.clbits)

    def num_tensor_factors(self):
        if False:
            while True:
                i = 10
        'Compute how many components the circuit can decompose into.'
        return rx.number_weakly_connected_components(self._multi_graph)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        try:
            self_phase = float(self.global_phase)
            other_phase = float(other.global_phase)
            if abs((self_phase - other_phase + np.pi) % (2 * np.pi) - np.pi) > 1e-10:
                return False
        except TypeError:
            if self.global_phase != other.global_phase:
                return False
        if self.calibrations != other.calibrations:
            return False
        self_bit_indices = {bit: idx for (idx, bit) in enumerate(self.qubits + self.clbits)}
        other_bit_indices = {bit: idx for (idx, bit) in enumerate(other.qubits + other.clbits)}
        self_qreg_indices = {regname: [self_bit_indices[bit] for bit in reg] for (regname, reg) in self.qregs.items()}
        self_creg_indices = {regname: [self_bit_indices[bit] for bit in reg] for (regname, reg) in self.cregs.items()}
        other_qreg_indices = {regname: [other_bit_indices[bit] for bit in reg] for (regname, reg) in other.qregs.items()}
        other_creg_indices = {regname: [other_bit_indices[bit] for bit in reg] for (regname, reg) in other.cregs.items()}
        if self_qreg_indices != other_qreg_indices or self_creg_indices != other_creg_indices:
            return False

        def node_eq(node_self, node_other):
            if False:
                i = 10
                return i + 15
            return DAGNode.semantic_eq(node_self, node_other, self_bit_indices, other_bit_indices)
        return rx.is_isomorphic_node_match(self._multi_graph, other._multi_graph, node_eq)

    def topological_nodes(self, key=None):
        if False:
            print('Hello World!')
        '\n        Yield nodes in topological order.\n\n        Args:\n            key (Callable): A callable which will take a DAGNode object and\n                return a string sort key. If not specified the\n                :attr:`~qiskit.dagcircuit.DAGNode.sort_key` attribute will be\n                used as the sort key for each node.\n\n        Returns:\n            generator(DAGOpNode, DAGInNode, or DAGOutNode): node in topological order\n        '

        def _key(x):
            if False:
                i = 10
                return i + 15
            return x.sort_key
        if key is None:
            key = _key
        return iter(rx.lexicographical_topological_sort(self._multi_graph, key=key))

    def topological_op_nodes(self, key=None) -> Generator[DAGOpNode, Any, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Yield op nodes in topological order.\n\n        Allowed to pass in specific key to break ties in top order\n\n        Args:\n            key (Callable): A callable which will take a DAGNode object and\n                return a string sort key. If not specified the\n                :attr:`~qiskit.dagcircuit.DAGNode.sort_key` attribute will be\n                used as the sort key for each node.\n\n        Returns:\n            generator(DAGOpNode): op node in topological order\n        '
        return (nd for nd in self.topological_nodes(key) if isinstance(nd, DAGOpNode))

    def replace_block_with_op(self, node_block, op, wire_pos_map, cycle_check=True):
        if False:
            return 10
        "Replace a block of nodes with a single node.\n\n        This is used to consolidate a block of DAGOpNodes into a single\n        operation. A typical example is a block of gates being consolidated\n        into a single ``UnitaryGate`` representing the unitary matrix of the\n        block.\n\n        Args:\n            node_block (List[DAGNode]): A list of dag nodes that represents the\n                node block to be replaced\n            op (qiskit.circuit.Operation): The operation to replace the\n                block with\n            wire_pos_map (Dict[Bit, int]): The dictionary mapping the bits to their positions in the\n                output ``qargs`` or ``cargs``. This is necessary to reconstruct the arg order over\n                multiple gates in the combined single op node.  If a :class:`.Bit` is not in the\n                dictionary, it will not be added to the args; this can be useful when dealing with\n                control-flow operations that have inherent bits in their ``condition`` or ``target``\n                fields.\n            cycle_check (bool): When set to True this method will check that\n                replacing the provided ``node_block`` with a single node\n                would introduce a cycle (which would invalidate the\n                ``DAGCircuit``) and will raise a ``DAGCircuitError`` if a cycle\n                would be introduced. This checking comes with a run time\n                penalty. If you can guarantee that your input ``node_block`` is\n                a contiguous block and won't introduce a cycle when it's\n                contracted to a single node, this can be set to ``False`` to\n                improve the runtime performance of this method.\n\n        Raises:\n            DAGCircuitError: if ``cycle_check`` is set to ``True`` and replacing\n                the specified block introduces a cycle or if ``node_block`` is\n                empty.\n\n        Returns:\n            DAGOpNode: The op node that replaces the block.\n        "
        block_qargs = set()
        block_cargs = set()
        block_ids = [x._node_id for x in node_block]
        if not node_block:
            raise DAGCircuitError("Can't replace an empty node_block")
        for nd in node_block:
            block_qargs |= set(nd.qargs)
            block_cargs |= set(nd.cargs)
            if (condition := getattr(nd.op, 'condition', None)) is not None:
                block_cargs.update(condition_resources(condition).clbits)
            elif isinstance(nd.op, SwitchCaseOp):
                if isinstance(nd.op.target, Clbit):
                    block_cargs.add(nd.op.target)
                elif isinstance(nd.op.target, ClassicalRegister):
                    block_cargs.update(nd.op.target)
                else:
                    block_cargs.update(node_resources(nd.op.target).clbits)
        block_qargs = [bit for bit in block_qargs if bit in wire_pos_map]
        block_qargs.sort(key=wire_pos_map.get)
        block_cargs = [bit for bit in block_cargs if bit in wire_pos_map]
        block_cargs.sort(key=wire_pos_map.get)
        new_node = DAGOpNode(op, block_qargs, block_cargs, dag=self)
        try:
            new_node._node_id = self._multi_graph.contract_nodes(block_ids, new_node, check_cycle=cycle_check)
        except rx.DAGWouldCycle as ex:
            raise DAGCircuitError('Replacing the specified node block would introduce a cycle') from ex
        self._increment_op(op)
        for nd in node_block:
            self._decrement_op(nd.op)
        return new_node

    def substitute_node_with_dag(self, node, input_dag, wires=None, propagate_condition=True):
        if False:
            return 10
        'Replace one node with dag.\n\n        Args:\n            node (DAGOpNode): node to substitute\n            input_dag (DAGCircuit): circuit that will substitute the node\n            wires (list[Bit] | Dict[Bit, Bit]): gives an order for (qu)bits\n                in the input circuit. If a list, then the bits refer to those in the ``input_dag``,\n                and the order gets matched to the node wires by qargs first, then cargs, then\n                conditions.  If a dictionary, then a mapping of bits in the ``input_dag`` to those\n                that the ``node`` acts on.\n            propagate_condition (bool): If ``True`` (default), then any ``condition`` attribute on\n                the operation within ``node`` is propagated to each node in the ``input_dag``.  If\n                ``False``, then the ``input_dag`` is assumed to faithfully implement suitable\n                conditional logic already.  This is ignored for :class:`.ControlFlowOp`\\ s (i.e.\n                treated as if it is ``False``); replacements of those must already fulfill the same\n                conditional logic or this function would be close to useless for them.\n\n        Returns:\n            dict: maps node IDs from `input_dag` to their new node incarnations in `self`.\n\n        Raises:\n            DAGCircuitError: if met with unexpected predecessor/successors\n        '
        if not isinstance(node, DAGOpNode):
            raise DAGCircuitError(f'expected node DAGOpNode, got {type(node)}')
        if isinstance(wires, dict):
            wire_map = wires
        else:
            wires = input_dag.wires if wires is None else wires
            node_cargs = set(node.cargs)
            node_wire_order = list(node.qargs) + list(node.cargs)
            if not propagate_condition and self._operation_may_have_bits(node.op):
                node_wire_order += [bit for bit in self._bits_in_operation(node.op) if bit not in node_cargs]
            if len(wires) != len(node_wire_order):
                raise DAGCircuitError(f'bit mapping invalid: expected {len(node_wire_order)}, got {len(wires)}')
            wire_map = dict(zip(wires, node_wire_order))
            if len(wire_map) != len(node_wire_order):
                raise DAGCircuitError('bit mapping invalid: some bits have duplicate entries')
        for (input_dag_wire, our_wire) in wire_map.items():
            if our_wire not in self.input_map:
                raise DAGCircuitError(f'bit mapping invalid: {our_wire} is not in this DAG')
            check_type = Qubit if isinstance(our_wire, Qubit) else Clbit
            if not isinstance(input_dag_wire, check_type):
                raise DAGCircuitError(f'bit mapping invalid: {input_dag_wire} and {our_wire} are different bit types')
        reverse_wire_map = {b: a for (a, b) in wire_map.items()}
        if propagate_condition and (not isinstance(node.op, ControlFlowOp)) and ((op_condition := getattr(node.op, 'condition', None)) is not None):
            in_dag = input_dag.copy_empty_like()
            (target, value) = op_condition
            if isinstance(target, Clbit):
                new_target = reverse_wire_map.get(target, Clbit())
                if new_target not in wire_map:
                    in_dag.add_clbits([new_target])
                    (wire_map[new_target], reverse_wire_map[target]) = (target, new_target)
                target_cargs = {new_target}
            else:
                mapped_bits = [reverse_wire_map.get(bit, Clbit()) for bit in target]
                for (ours, theirs) in zip(target, mapped_bits):
                    (wire_map[theirs], reverse_wire_map[ours]) = (ours, theirs)
                new_target = ClassicalRegister(bits=mapped_bits)
                in_dag.add_creg(new_target)
                target_cargs = set(new_target)
            new_condition = (new_target, value)
            for in_node in input_dag.topological_op_nodes():
                if getattr(in_node.op, 'condition', None) is not None:
                    raise DAGCircuitError('cannot propagate a condition to an element that already has one')
                if target_cargs.intersection(in_node.cargs):
                    raise DAGCircuitError('cannot propagate a condition to an element that acts on those bits')
                new_op = copy.copy(in_node.op)
                if new_condition:
                    if not isinstance(new_op, ControlFlowOp):
                        new_op = new_op.c_if(*new_condition)
                    else:
                        new_op.condition = new_condition
                in_dag.apply_operation_back(new_op, in_node.qargs, in_node.cargs, check=False)
        else:
            in_dag = input_dag
        if in_dag.global_phase:
            self.global_phase += in_dag.global_phase
        for (in_dag_wire, self_wire) in wire_map.items():
            input_node = in_dag.input_map[in_dag_wire]
            output_node = in_dag.output_map[in_dag_wire]
            if in_dag._multi_graph.has_edge(input_node._node_id, output_node._node_id):
                pred = self._multi_graph.find_predecessors_by_edge(node._node_id, lambda edge, wire=self_wire: edge == wire)[0]
                succ = self._multi_graph.find_successors_by_edge(node._node_id, lambda edge, wire=self_wire: edge == wire)[0]
                self._multi_graph.add_edge(pred._node_id, succ._node_id, self_wire)

        def filter_fn(node):
            if False:
                return 10
            if not isinstance(node, DAGOpNode):
                return False
            for qarg in node.qargs:
                if qarg not in wire_map:
                    return False
            return True

        def edge_map_fn(source, _target, self_wire):
            if False:
                for i in range(10):
                    print('nop')
            wire = reverse_wire_map[self_wire]
            if source == node._node_id:
                wire_output_id = in_dag.output_map[wire]._node_id
                out_index = in_dag._multi_graph.predecessor_indices(wire_output_id)[0]
                if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
                    return None
            else:
                wire_input_id = in_dag.input_map[wire]._node_id
                out_index = in_dag._multi_graph.successor_indices(wire_input_id)[0]
                if not isinstance(in_dag._multi_graph[out_index], DAGOpNode):
                    return None
            return out_index

        def edge_weight_map(wire):
            if False:
                while True:
                    i = 10
            return wire_map[wire]
        node_map = self._multi_graph.substitute_node_with_subgraph(node._node_id, in_dag._multi_graph, edge_map_fn, filter_fn, edge_weight_map)
        self._decrement_op(node.op)
        variable_mapper = _classical_resource_map.VariableMapper(self.cregs.values(), wire_map, self.add_creg)
        for (old_node_index, new_node_index) in node_map.items():
            old_node = in_dag._multi_graph[old_node_index]
            if isinstance(old_node.op, SwitchCaseOp):
                m_op = SwitchCaseOp(variable_mapper.map_target(old_node.op.target), old_node.op.cases_specifier(), label=old_node.op.label)
            elif getattr(old_node.op, 'condition', None) is not None:
                m_op = old_node.op
                if not isinstance(old_node.op, ControlFlowOp):
                    new_condition = variable_mapper.map_condition(m_op.condition)
                    if new_condition is not None:
                        m_op = m_op.c_if(*new_condition)
                else:
                    m_op.condition = variable_mapper.map_condition(m_op.condition)
            else:
                m_op = old_node.op
            m_qargs = [wire_map[x] for x in old_node.qargs]
            m_cargs = [wire_map[x] for x in old_node.cargs]
            new_node = DAGOpNode(m_op, qargs=m_qargs, cargs=m_cargs, dag=self)
            new_node._node_id = new_node_index
            self._multi_graph[new_node_index] = new_node
            self._increment_op(new_node.op)
        return {k: self._multi_graph[v] for (k, v) in node_map.items()}

    def substitute_node(self, node, op, inplace=False, propagate_condition=True):
        if False:
            print('Hello World!')
        'Replace an DAGOpNode with a single operation. qargs, cargs and\n        conditions for the new operation will be inferred from the node to be\n        replaced. The new operation will be checked to match the shape of the\n        replaced operation.\n\n        Args:\n            node (DAGOpNode): Node to be replaced\n            op (qiskit.circuit.Operation): The :class:`qiskit.circuit.Operation`\n                instance to be added to the DAG\n            inplace (bool): Optional, default False. If True, existing DAG node\n                will be modified to include op. Otherwise, a new DAG node will\n                be used.\n            propagate_condition (bool): Optional, default True.  If True, a condition on the\n                ``node`` to be replaced will be applied to the new ``op``.  This is the legacy\n                behaviour.  If either node is a control-flow operation, this will be ignored.  If\n                the ``op`` already has a condition, :exc:`.DAGCircuitError` is raised.\n\n        Returns:\n            DAGOpNode: the new node containing the added operation.\n\n        Raises:\n            DAGCircuitError: If replacement operation was incompatible with\n            location of target node.\n        '
        if not isinstance(node, DAGOpNode):
            raise DAGCircuitError('Only DAGOpNodes can be replaced.')
        if node.op.num_qubits != op.num_qubits or node.op.num_clbits != op.num_clbits:
            raise DAGCircuitError('Cannot replace node of width ({} qubits, {} clbits) with operation of mismatched width ({} qubits, {} clbits).'.format(node.op.num_qubits, node.op.num_clbits, op.num_qubits, op.num_clbits))
        current_wires = {wire for (_, _, wire) in self.edges(node)}
        new_wires = set(node.qargs) | set(node.cargs)
        if (new_condition := getattr(op, 'condition', None)) is not None:
            new_wires.update(condition_resources(new_condition).clbits)
        elif isinstance(op, SwitchCaseOp):
            if isinstance(op.target, Clbit):
                new_wires.add(op.target)
            elif isinstance(op.target, ClassicalRegister):
                new_wires.update(op.target)
            else:
                new_wires.update(node_resources(op.target).clbits)
        if propagate_condition and (not (isinstance(node.op, ControlFlowOp) or isinstance(op, ControlFlowOp))):
            if new_condition is not None:
                raise DAGCircuitError('Cannot propagate a condition to an operation that already has one.')
            if (old_condition := getattr(node.op, 'condition', None)) is not None:
                if not isinstance(op, Instruction):
                    raise DAGCircuitError('Cannot add a condition on a generic Operation.')
                if not isinstance(node.op, ControlFlowOp):
                    op = op.c_if(*old_condition)
                else:
                    op.condition = old_condition
                new_wires.update(condition_resources(old_condition).clbits)
        if new_wires != current_wires:
            raise DAGCircuitError(f"New operation '{op}' does not span the same wires as the old node '{node}'. New wires: {new_wires}, old wires: {current_wires}.")
        if inplace:
            if op.name != node.op.name:
                self._increment_op(op)
                self._decrement_op(node.op)
            node.op = op
            return node
        new_node = copy.copy(node)
        new_node.op = op
        self._multi_graph[node._node_id] = new_node
        if op.name != node.op.name:
            self._increment_op(op)
            self._decrement_op(node.op)
        return new_node

    def separable_circuits(self, remove_idle_qubits=False) -> List['DAGCircuit']:
        if False:
            for i in range(10):
                print('nop')
        'Decompose the circuit into sets of qubits with no gates connecting them.\n\n        Args:\n            remove_idle_qubits (bool): Flag denoting whether to remove idle qubits from\n                the separated circuits. If ``False``, each output circuit will contain the\n                same number of qubits as ``self``.\n\n        Returns:\n            List[DAGCircuit]: The circuits resulting from separating ``self`` into sets\n                of disconnected qubits\n\n        Each :class:`~.DAGCircuit` instance returned by this method will contain the same number of\n        clbits as ``self``. The global phase information in ``self`` will not be maintained\n        in the subcircuits returned by this method.\n        '
        connected_components = rx.weakly_connected_components(self._multi_graph)
        disconnected_subgraphs = []
        for components in connected_components:
            disconnected_subgraphs.append(self._multi_graph.subgraph(list(components)))

        def _key(x):
            if False:
                i = 10
                return i + 15
            return x.sort_key
        decomposed_dags = []
        for subgraph in disconnected_subgraphs:
            new_dag = self.copy_empty_like()
            new_dag.global_phase = 0
            subgraph_is_classical = True
            for node in rx.lexicographical_topological_sort(subgraph, key=_key):
                if isinstance(node, DAGInNode):
                    if isinstance(node.wire, Qubit):
                        subgraph_is_classical = False
                if not isinstance(node, DAGOpNode):
                    continue
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
            if not subgraph_is_classical:
                decomposed_dags.append(new_dag)
        if remove_idle_qubits:
            for dag in decomposed_dags:
                dag.remove_qubits(*(bit for bit in dag.idle_wires() if isinstance(bit, Qubit)))
        return decomposed_dags

    def swap_nodes(self, node1, node2):
        if False:
            return 10
        'Swap connected nodes e.g. due to commutation.\n\n        Args:\n            node1 (OpNode): predecessor node\n            node2 (OpNode): successor node\n\n        Raises:\n            DAGCircuitError: if either node is not an OpNode or nodes are not connected\n        '
        if not (isinstance(node1, DAGOpNode) and isinstance(node2, DAGOpNode)):
            raise DAGCircuitError('nodes to swap are not both DAGOpNodes')
        try:
            connected_edges = self._multi_graph.get_all_edge_data(node1._node_id, node2._node_id)
        except rx.NoEdgeBetweenNodes as no_common_edge:
            raise DAGCircuitError('attempt to swap unconnected nodes') from no_common_edge
        node1_id = node1._node_id
        node2_id = node2._node_id
        for edge in connected_edges[::-1]:
            edge_find = lambda x, y=edge: x == y
            edge_parent = self._multi_graph.find_predecessors_by_edge(node1_id, edge_find)[0]
            self._multi_graph.remove_edge(edge_parent._node_id, node1_id)
            self._multi_graph.add_edge(edge_parent._node_id, node2_id, edge)
            edge_child = self._multi_graph.find_successors_by_edge(node2_id, edge_find)[0]
            self._multi_graph.remove_edge(node1_id, node2_id)
            self._multi_graph.add_edge(node2_id, node1_id, edge)
            self._multi_graph.remove_edge(node2_id, edge_child._node_id)
            self._multi_graph.add_edge(node1_id, edge_child._node_id, edge)

    def node(self, node_id):
        if False:
            return 10
        'Get the node in the dag.\n\n        Args:\n            node_id(int): Node identifier.\n\n        Returns:\n            node: the node.\n        '
        return self._multi_graph[node_id]

    def nodes(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterator for node values.\n\n        Yield:\n            node: the node.\n        '
        yield from self._multi_graph.nodes()

    def edges(self, nodes=None):
        if False:
            while True:
                i = 10
        'Iterator for edge values and source and dest node\n\n        This works by returning the output edges from the specified nodes. If\n        no nodes are specified all edges from the graph are returned.\n\n        Args:\n            nodes(DAGOpNode, DAGInNode, or DAGOutNode|list(DAGOpNode, DAGInNode, or DAGOutNode):\n                Either a list of nodes or a single input node. If none is specified,\n                all edges are returned from the graph.\n\n        Yield:\n            edge: the edge in the same format as out_edges the tuple\n                (source node, destination node, edge data)\n        '
        if nodes is None:
            nodes = self._multi_graph.nodes()
        elif isinstance(nodes, (DAGOpNode, DAGInNode, DAGOutNode)):
            nodes = [nodes]
        for node in nodes:
            raw_nodes = self._multi_graph.out_edges(node._node_id)
            for (source, dest, edge) in raw_nodes:
                yield (self._multi_graph[source], self._multi_graph[dest], edge)

    def op_nodes(self, op=None, include_directives=True):
        if False:
            i = 10
            return i + 15
        'Get the list of "op" nodes in the dag.\n\n        Args:\n            op (Type): :class:`qiskit.circuit.Operation` subclass op nodes to\n                return. If None, return all op nodes.\n            include_directives (bool): include `barrier`, `snapshot` etc.\n\n        Returns:\n            list[DAGOpNode]: the list of node ids containing the given op.\n        '
        nodes = []
        for node in self._multi_graph.nodes():
            if isinstance(node, DAGOpNode):
                if not include_directives and getattr(node.op, '_directive', False):
                    continue
                if op is None or isinstance(node.op, op):
                    nodes.append(node)
        return nodes

    def gate_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the list of gate nodes in the dag.\n\n        Returns:\n            list[DAGOpNode]: the list of DAGOpNodes that represent gates.\n        '
        nodes = []
        for node in self.op_nodes():
            if isinstance(node.op, Gate):
                nodes.append(node)
        return nodes

    def named_nodes(self, *names):
        if False:
            return 10
        'Get the set of "op" nodes with the given name.'
        named_nodes = []
        for node in self._multi_graph.nodes():
            if isinstance(node, DAGOpNode) and node.op.name in names:
                named_nodes.append(node)
        return named_nodes

    def two_qubit_ops(self):
        if False:
            i = 10
            return i + 15
        'Get list of 2 qubit operations. Ignore directives like snapshot and barrier.'
        ops = []
        for node in self.op_nodes(include_directives=False):
            if len(node.qargs) == 2:
                ops.append(node)
        return ops

    def multi_qubit_ops(self):
        if False:
            for i in range(10):
                print('nop')
        'Get list of 3+ qubit operations. Ignore directives like snapshot and barrier.'
        ops = []
        for node in self.op_nodes(include_directives=False):
            if len(node.qargs) >= 3:
                ops.append(node)
        return ops

    def longest_path(self):
        if False:
            i = 10
            return i + 15
        'Returns the longest path in the dag as a list of DAGOpNodes, DAGInNodes, and DAGOutNodes.'
        return [self._multi_graph[x] for x in rx.dag_longest_path(self._multi_graph)]

    def successors(self, node):
        if False:
            print('Hello World!')
        'Returns iterator of the successors of a node as DAGOpNodes and DAGOutNodes.'
        return iter(self._multi_graph.successors(node._node_id))

    def predecessors(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Returns iterator of the predecessors of a node as DAGOpNodes and DAGInNodes.'
        return iter(self._multi_graph.predecessors(node._node_id))

    def is_successor(self, node, node_succ):
        if False:
            print('Hello World!')
        'Checks if a second node is in the successors of node.'
        return self._multi_graph.has_edge(node._node_id, node_succ._node_id)

    def is_predecessor(self, node, node_pred):
        if False:
            for i in range(10):
                print('nop')
        'Checks if a second node is in the predecessors of node.'
        return self._multi_graph.has_edge(node_pred._node_id, node._node_id)

    def quantum_predecessors(self, node):
        if False:
            i = 10
            return i + 15
        'Returns iterator of the predecessors of a node that are\n        connected by a quantum edge as DAGOpNodes and DAGInNodes.'
        return iter(self._multi_graph.find_predecessors_by_edge(node._node_id, lambda edge_data: isinstance(edge_data, Qubit)))

    def classical_predecessors(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Returns iterator of the predecessors of a node that are\n        connected by a classical edge as DAGOpNodes and DAGInNodes.'
        return iter(self._multi_graph.find_predecessors_by_edge(node._node_id, lambda edge_data: isinstance(edge_data, Clbit)))

    def ancestors(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Returns set of the ancestors of a node as DAGOpNodes and DAGInNodes.'
        return {self._multi_graph[x] for x in rx.ancestors(self._multi_graph, node._node_id)}

    def descendants(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Returns set of the descendants of a node as DAGOpNodes and DAGOutNodes.'
        return {self._multi_graph[x] for x in rx.descendants(self._multi_graph, node._node_id)}

    def bfs_successors(self, node):
        if False:
            while True:
                i = 10
        '\n        Returns an iterator of tuples of (DAGNode, [DAGNodes]) where the DAGNode is the current node\n        and [DAGNode] is its successors in  BFS order.\n        '
        return iter(rx.bfs_successors(self._multi_graph, node._node_id))

    def quantum_successors(self, node):
        if False:
            i = 10
            return i + 15
        'Returns iterator of the successors of a node that are\n        connected by a quantum edge as Opnodes and DAGOutNodes.'
        return iter(self._multi_graph.find_successors_by_edge(node._node_id, lambda edge_data: isinstance(edge_data, Qubit)))

    def classical_successors(self, node):
        if False:
            return 10
        'Returns iterator of the successors of a node that are\n        connected by a classical edge as DAGOpNodes and DAGInNodes.'
        return iter(self._multi_graph.find_successors_by_edge(node._node_id, lambda edge_data: isinstance(edge_data, Clbit)))

    def remove_op_node(self, node):
        if False:
            return 10
        'Remove an operation node n.\n\n        Add edges from predecessors to successors.\n        '
        if not isinstance(node, DAGOpNode):
            raise DAGCircuitError('The method remove_op_node only works on DAGOpNodes. A "%s" node type was wrongly provided.' % type(node))
        self._multi_graph.remove_node_retain_edges(node._node_id, use_outgoing=False, condition=lambda edge1, edge2: edge1 == edge2)
        self._decrement_op(node.op)

    def remove_ancestors_of(self, node):
        if False:
            return 10
        'Remove all of the ancestor operation nodes of node.'
        anc = rx.ancestors(self._multi_graph, node)
        for anc_node in anc:
            if isinstance(anc_node, DAGOpNode):
                self.remove_op_node(anc_node)

    def remove_descendants_of(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Remove all of the descendant operation nodes of node.'
        desc = rx.descendants(self._multi_graph, node)
        for desc_node in desc:
            if isinstance(desc_node, DAGOpNode):
                self.remove_op_node(desc_node)

    def remove_nonancestors_of(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Remove all of the non-ancestors operation nodes of node.'
        anc = rx.ancestors(self._multi_graph, node)
        comp = list(set(self._multi_graph.nodes()) - set(anc))
        for n in comp:
            if isinstance(n, DAGOpNode):
                self.remove_op_node(n)

    def remove_nondescendants_of(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Remove all of the non-descendants operation nodes of node.'
        dec = rx.descendants(self._multi_graph, node)
        comp = list(set(self._multi_graph.nodes()) - set(dec))
        for n in comp:
            if isinstance(n, DAGOpNode):
                self.remove_op_node(n)

    def front_layer(self):
        if False:
            i = 10
            return i + 15
        'Return a list of op nodes in the first layer of this dag.'
        graph_layers = self.multigraph_layers()
        try:
            next(graph_layers)
        except StopIteration:
            return []
        op_nodes = [node for node in next(graph_layers) if isinstance(node, DAGOpNode)]
        return op_nodes

    def layers(self):
        if False:
            i = 10
            return i + 15
        'Yield a shallow view on a layer of this DAGCircuit for all d layers of this circuit.\n\n        A layer is a circuit whose gates act on disjoint qubits, i.e.,\n        a layer has depth 1. The total number of layers equals the\n        circuit depth d. The layers are indexed from 0 to d-1 with the\n        earliest layer at index 0. The layers are constructed using a\n        greedy algorithm. Each returned layer is a dict containing\n        {"graph": circuit graph, "partition": list of qubit lists}.\n\n        The returned layer contains new (but semantically equivalent) DAGOpNodes, DAGInNodes,\n        and DAGOutNodes. These are not the same as nodes of the original dag, but are equivalent\n        via DAGNode.semantic_eq(node1, node2).\n\n        TODO: Gates that use the same cbits will end up in different\n        layers as this is currently implemented. This may not be\n        the desired behavior.\n        '
        graph_layers = self.multigraph_layers()
        try:
            next(graph_layers)
        except StopIteration:
            return
        for graph_layer in graph_layers:
            op_nodes = [node for node in graph_layer if isinstance(node, DAGOpNode)]
            op_nodes.sort(key=lambda nd: nd._node_id)
            if not op_nodes:
                return
            new_layer = self.copy_empty_like()
            for node in op_nodes:
                new_layer.apply_operation_back(node.op, node.qargs, node.cargs, check=False)
            support_list = [op_node.qargs for op_node in new_layer.op_nodes() if not getattr(op_node.op, '_directive', False)]
            yield {'graph': new_layer, 'partition': support_list}

    def serial_layers(self):
        if False:
            return 10
        'Yield a layer for all gates of this circuit.\n\n        A serial layer is a circuit with one gate. The layers have the\n        same structure as in layers().\n        '
        for next_node in self.topological_op_nodes():
            new_layer = self.copy_empty_like()
            support_list = []
            op = copy.copy(next_node.op)
            qargs = copy.copy(next_node.qargs)
            cargs = copy.copy(next_node.cargs)
            new_layer.apply_operation_back(op, qargs, cargs, check=False)
            if not getattr(next_node.op, '_directive', False):
                support_list.append(list(qargs))
            l_dict = {'graph': new_layer, 'partition': support_list}
            yield l_dict

    def multigraph_layers(self):
        if False:
            i = 10
            return i + 15
        'Yield layers of the multigraph.'
        first_layer = [x._node_id for x in self.input_map.values()]
        return iter(rx.layers(self._multi_graph, first_layer))

    def collect_runs(self, namelist):
        if False:
            i = 10
            return i + 15
        'Return a set of non-conditional runs of "op" nodes with the given names.\n\n        For example, "... h q[0]; cx q[0],q[1]; cx q[0],q[1]; h q[1]; .."\n        would produce the tuple of cx nodes as an element of the set returned\n        from a call to collect_runs(["cx"]). If instead the cx nodes were\n        "cx q[0],q[1]; cx q[1],q[0];", the method would still return the\n        pair in a tuple. The namelist can contain names that are not\n        in the circuit\'s basis.\n\n        Nodes must have only one successor to continue the run.\n        '

        def filter_fn(node):
            if False:
                while True:
                    i = 10
            return isinstance(node, DAGOpNode) and node.op.name in namelist and (getattr(node.op, 'condition', None) is None)
        group_list = rx.collect_runs(self._multi_graph, filter_fn)
        return {tuple(x) for x in group_list}

    def collect_1q_runs(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a set of non-conditional runs of 1q "op" nodes.'

        def filter_fn(node):
            if False:
                print('Hello World!')
            return isinstance(node, DAGOpNode) and len(node.qargs) == 1 and (len(node.cargs) == 0) and (getattr(node.op, 'condition', None) is None) and (not node.op.is_parameterized()) and isinstance(node.op, Gate) and hasattr(node.op, '__array__')
        return rx.collect_runs(self._multi_graph, filter_fn)

    def collect_2q_runs(self):
        if False:
            print('Hello World!')
        'Return a set of non-conditional runs of 2q "op" nodes.'
        to_qid = {}
        for (i, qubit) in enumerate(self.qubits):
            to_qid[qubit] = i

        def filter_fn(node):
            if False:
                while True:
                    i = 10
            if isinstance(node, DAGOpNode):
                return isinstance(node.op, Gate) and len(node.qargs) <= 2 and (not getattr(node.op, 'condition', None)) and (not node.op.is_parameterized())
            else:
                return None

        def color_fn(edge):
            if False:
                while True:
                    i = 10
            if isinstance(edge, Qubit):
                return to_qid[edge]
            else:
                return None
        return rx.collect_bicolor_runs(self._multi_graph, filter_fn, color_fn)

    def nodes_on_wire(self, wire, only_ops=False):
        if False:
            while True:
                i = 10
        "\n        Iterator for nodes that affect a given wire.\n\n        Args:\n            wire (Bit): the wire to be looked at.\n            only_ops (bool): True if only the ops nodes are wanted;\n                        otherwise, all nodes are returned.\n        Yield:\n             Iterator: the successive nodes on the given wire\n\n        Raises:\n            DAGCircuitError: if the given wire doesn't exist in the DAG\n        "
        current_node = self.input_map.get(wire, None)
        if not current_node:
            raise DAGCircuitError('The given wire %s is not present in the circuit' % str(wire))
        more_nodes = True
        while more_nodes:
            more_nodes = False
            if isinstance(current_node, DAGOpNode) or not only_ops:
                yield current_node
            try:
                current_node = self._multi_graph.find_adjacent_node_by_edge(current_node._node_id, lambda x: wire == x)
                more_nodes = True
            except rx.NoSuitableNeighbors:
                pass

    def count_ops(self, *, recurse: bool=True):
        if False:
            i = 10
            return i + 15
        'Count the occurrences of operation names.\n\n        Args:\n            recurse: if ``True`` (default), then recurse into control-flow operations.  In all\n                cases, this counts only the number of times the operation appears in any possible\n                block; both branches of if-elses are counted, and for- and while-loop blocks are\n                only counted once.\n\n        Returns:\n            Mapping[str, int]: a mapping of operation names to the number of times it appears.\n        '
        if not recurse or not CONTROL_FLOW_OP_NAMES.intersection(self._op_names):
            return self._op_names.copy()
        from qiskit.converters import circuit_to_dag

        def inner(dag, counts):
            if False:
                for i in range(10):
                    print('nop')
            for (name, count) in dag._op_names.items():
                counts[name] += count
            for node in dag.op_nodes(ControlFlowOp):
                for block in node.op.blocks:
                    counts = inner(circuit_to_dag(block), counts)
            return counts
        return dict(inner(self, defaultdict(int)))

    def count_ops_longest_path(self):
        if False:
            return 10
        'Count the occurrences of operation names on the longest path.\n\n        Returns a dictionary of counts keyed on the operation name.\n        '
        op_dict = {}
        path = self.longest_path()
        path = path[1:-1]
        for node in path:
            name = node.op.name
            if name not in op_dict:
                op_dict[name] = 1
            else:
                op_dict[name] += 1
        return op_dict

    def quantum_causal_cone(self, qubit):
        if False:
            while True:
                i = 10
        "\n        Returns causal cone of a qubit.\n\n        A qubit's causal cone is the set of qubits that can influence the output of that\n        qubit through interactions, whether through multi-qubit gates or operations. Knowing\n        the causal cone of a qubit can be useful when debugging faulty circuits, as it can\n        help identify which wire(s) may be causing the problem.\n\n        This method does not consider any classical data dependency in the ``DAGCircuit``,\n        classical bit wires are ignored for the purposes of building the causal cone.\n\n        Args:\n            qubit (~qiskit.circuit.Qubit): The output qubit for which we want to find the causal cone.\n\n        Returns:\n            Set[~qiskit.circuit.Qubit]: The set of qubits whose interactions affect ``qubit``.\n        "
        output_node = self.output_map.get(qubit, None)
        if not output_node:
            raise DAGCircuitError(f'Qubit {qubit} is not part of this circuit.')
        qubits_to_check = {qubit}
        queue = deque(self.predecessors(output_node))
        while queue:
            node_to_check = queue.popleft()
            if isinstance(node_to_check, DAGOpNode):
                qubit_set = set(node_to_check.qargs)
                if len(qubit_set.intersection(qubits_to_check)) > 0 and node_to_check.op.name != 'barrier' and (not getattr(node_to_check.op, '_directive')):
                    qubits_to_check = qubits_to_check.union(qubit_set)
            for node in self.quantum_predecessors(node_to_check):
                if isinstance(node, DAGOpNode) and len(qubits_to_check.intersection(set(node.qargs))) > 0:
                    queue.append(node)
        return qubits_to_check

    def properties(self):
        if False:
            return 10
        'Return a dictionary of circuit properties.'
        summary = {'size': self.size(), 'depth': self.depth(), 'width': self.width(), 'qubits': self.num_qubits(), 'bits': self.num_clbits(), 'factors': self.num_tensor_factors(), 'operations': self.count_ops()}
        return summary

    def draw(self, scale=0.7, filename=None, style='color'):
        if False:
            print('Hello World!')
        "\n        Draws the dag circuit.\n\n        This function needs `pydot <https://github.com/erocarrera/pydot>`_, which in turn needs\n        `Graphviz <https://www.graphviz.org/>`_ to be installed.\n\n        Args:\n            scale (float): scaling factor\n            filename (str): file path to save image to (format inferred from name)\n            style (str):\n                'plain': B&W graph;\n                'color' (default): color input/output/op nodes\n\n        Returns:\n            Ipython.display.Image: if in Jupyter notebook and not saving to file,\n            otherwise None.\n        "
        from qiskit.visualization.dag_visualization import dag_drawer
        return dag_drawer(dag=self, scale=scale, filename=filename, style=style)
"""Padding pass to fill empty timeslot."""
from __future__ import annotations
from collections.abc import Iterable
import logging
from qiskit.circuit import Qubit, Clbit, Instruction
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
logger = logging.getLogger(__name__)

class BasePadding(TransformationPass):
    """The base class of padding pass.

    This pass requires one of scheduling passes to be executed before itself.
    Since there are multiple scheduling strategies, the selection of scheduling
    pass is left in the hands of the pass manager designer.
    Once a scheduling analysis pass is run, ``node_start_time`` is generated
    in the :attr:`property_set`.  This information is represented by a python dictionary of
    the expected instruction execution times keyed on the node instances.
    Entries in the dictionary are only created for non-delay nodes.
    The padding pass expects all ``DAGOpNode`` in the circuit to be scheduled.

    This base class doesn't define any sequence to interleave, but it manages
    the location where the sequence is inserted, and provides a set of information necessary
    to construct the proper sequence. Thus, a subclass of this pass just needs to implement
    :meth:`_pad` method, in which the subclass constructs a circuit block to insert.
    This mechanism removes lots of boilerplate logic to manage whole DAG circuits.

    Note that padding pass subclasses should define interleaving sequences satisfying:

        - Interleaved sequence does not change start time of other nodes
        - Interleaved sequence should have total duration of the provided ``time_interval``.

    Any manipulation violating these constraints may prevent this base pass from correctly
    tracking the start time of each instruction,
    which may result in violation of hardware alignment constraints.
    """

    def __init__(self, target: Target=None):
        if False:
            i = 10
            return i + 15
        'BasePadding initializer.\n\n        Args:\n            target: The :class:`~.Target` representing the target backend.\n                If it supplied and it does not support delay instruction on a qubit,\n                padding passes do not pad any idle time of the qubit.\n        '
        super().__init__()
        self.target = target

    def run(self, dag: DAGCircuit):
        if False:
            return 10
        'Run the padding pass on ``dag``.\n\n        Args:\n            dag: DAG to be checked.\n\n        Returns:\n            DAGCircuit: DAG with idle time filled with instructions.\n\n        Raises:\n            TranspilerError: When a particular node is not scheduled, likely some transform pass\n                is inserted before this node is called.\n        '
        self._pre_runhook(dag)
        node_start_time = self.property_set['node_start_time'].copy()
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        self.property_set['node_start_time'].clear()
        new_dag.name = dag.name
        new_dag.metadata = dag.metadata
        new_dag.unit = self.property_set['time_unit']
        new_dag.calibrations = dag.calibrations
        new_dag.global_phase = dag.global_phase
        idle_after = {bit: 0 for bit in dag.qubits}
        circuit_duration = 0
        for node in dag.topological_op_nodes():
            if node in node_start_time:
                t0 = node_start_time[node]
                t1 = t0 + node.op.duration
                circuit_duration = max(circuit_duration, t1)
                if isinstance(node.op, Delay):
                    dag.remove_op_node(node)
                    continue
                for bit in node.qargs:
                    if t0 - idle_after[bit] > 0 and self.__delay_supported(dag.find_bit(bit).index):
                        prev_node = next(new_dag.predecessors(new_dag.output_map[bit]))
                        self._pad(dag=new_dag, qubit=bit, t_start=idle_after[bit], t_end=t0, next_node=node, prev_node=prev_node)
                    idle_after[bit] = t1
                self._apply_scheduled_op(new_dag, t0, node.op, node.qargs, node.cargs)
            else:
                raise TranspilerError(f'Operation {repr(node)} is likely added after the circuit is scheduled. Schedule the circuit again if you transformed it.')
        for bit in new_dag.qubits:
            if circuit_duration - idle_after[bit] > 0 and self.__delay_supported(dag.find_bit(bit).index):
                node = new_dag.output_map[bit]
                prev_node = next(new_dag.predecessors(node))
                self._pad(dag=new_dag, qubit=bit, t_start=idle_after[bit], t_end=circuit_duration, next_node=node, prev_node=prev_node)
        new_dag.duration = circuit_duration
        return new_dag

    def __delay_supported(self, qarg: int) -> bool:
        if False:
            return 10
        'Delay operation is supported on the qubit (qarg) or not.'
        if self.target is None or self.target.instruction_supported('delay', qargs=(qarg,)):
            return True
        return False

    def _pre_runhook(self, dag: DAGCircuit):
        if False:
            i = 10
            return i + 15
        'Extra routine inserted before running the padding pass.\n\n        Args:\n            dag: DAG circuit on which the sequence is applied.\n\n        Raises:\n            TranspilerError: If the whole circuit or instruction is not scheduled.\n        '
        if 'node_start_time' not in self.property_set:
            raise TranspilerError(f'The input circuit {dag.name} is not scheduled. Call one of scheduling passes before running the {self.__class__.__name__} pass.')
        for (qarg, _) in enumerate(dag.qubits):
            if not self.__delay_supported(qarg):
                logger.debug('No padding on qubit %d as delay is not supported on it', qarg)

    def _apply_scheduled_op(self, dag: DAGCircuit, t_start: int, oper: Instruction, qubits: Qubit | Iterable[Qubit], clbits: Clbit | Iterable[Clbit]=()):
        if False:
            for i in range(10):
                print('nop')
        'Add new operation to DAG with scheduled information.\n\n        This is identical to apply_operation_back + updating the node_start_time propety.\n\n        Args:\n            dag: DAG circuit on which the sequence is applied.\n            t_start: Start time of new node.\n            oper: New operation that is added to the DAG circuit.\n            qubits: The list of qubits that the operation acts on.\n            clbits: The list of clbits that the operation acts on.\n        '
        if isinstance(qubits, Qubit):
            qubits = [qubits]
        if isinstance(clbits, Clbit):
            clbits = [clbits]
        new_node = dag.apply_operation_back(oper, qargs=qubits, cargs=clbits, check=False)
        self.property_set['node_start_time'][new_node] = t_start

    def _pad(self, dag: DAGCircuit, qubit: Qubit, t_start: int, t_end: int, next_node: DAGNode, prev_node: DAGNode):
        if False:
            print('Hello World!')
        "Interleave instruction sequence in between two nodes.\n\n        .. note::\n            If a DAGOpNode is added here, it should update node_start_time property\n            in the property set so that the added node is also scheduled.\n            This is achieved by adding operation via :meth:`_apply_scheduled_op`.\n\n        .. note::\n\n            This method doesn't check if the total duration of new DAGOpNode added here\n            is identical to the interval (``t_end - t_start``).\n            A developer of the pass must guarantee this is satisfied.\n            If the duration is greater than the interval, your circuit may be\n            compiled down to the target code with extra duration on the backend compiler,\n            which is then played normally without error. However, the outcome of your circuit\n            might be unexpected due to erroneous scheduling.\n\n        Args:\n            dag: DAG circuit that sequence is applied.\n            qubit: The wire that the sequence is applied on.\n            t_start: Absolute start time of this interval.\n            t_end: Absolute end time of this interval.\n            next_node: Node that follows the sequence.\n            prev_node: Node ahead of the sequence.\n        "
        raise NotImplementedError
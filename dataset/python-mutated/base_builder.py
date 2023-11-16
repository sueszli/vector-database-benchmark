"""Calibration builder base class."""
from abc import abstractmethod
from typing import List, Union
from qiskit.circuit import Instruction as CircuitInst
from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.calibration_entries import CalibrationPublisher
from qiskit.transpiler.basepasses import TransformationPass
from .exceptions import CalibrationNotAvailable

class CalibrationBuilder(TransformationPass):
    """Abstract base class to inject calibrations into circuits."""

    @abstractmethod
    def supported(self, node_op: CircuitInst, qubits: List) -> bool:
        if False:
            i = 10
            return i + 15
        'Determine if a given node supports the calibration.\n\n        Args:\n            node_op: Target instruction object.\n            qubits: Integer qubit indices to check.\n\n        Returns:\n            Return ``True`` is calibration can be provided.\n        '

    @abstractmethod
    def get_calibration(self, node_op: CircuitInst, qubits: List) -> Union[Schedule, ScheduleBlock]:
        if False:
            i = 10
            return i + 15
        'Gets the calibrated schedule for the given instruction and qubits.\n\n        Args:\n            node_op: Target instruction object.\n            qubits: Integer qubit indices to check.\n\n        Returns:\n            Return Schedule of target gate instruction.\n        '

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if False:
            return 10
        'Run the calibration adder pass on `dag`.\n\n        Args:\n            dag: DAG to schedule.\n\n        Returns:\n            A DAG with calibrations added to it.\n        '
        for node in dag.gate_nodes():
            qubits = [dag.find_bit(q).index for q in node.qargs]
            if self.supported(node.op, qubits) and (not dag.has_calibration_for(node)):
                try:
                    schedule = self.get_calibration(node.op, qubits)
                except CalibrationNotAvailable:
                    continue
                publisher = schedule.metadata.get('publisher', CalibrationPublisher.QISKIT)
                if publisher != CalibrationPublisher.BACKEND_PROVIDER:
                    dag.add_calibration(gate=node.op, qubits=qubits, schedule=schedule)
        return dag
"""Allocate all idle nodes from the coupling map as ancilla on the layout."""
from qiskit.circuit import QuantumRegister
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target

class FullAncillaAllocation(AnalysisPass):
    """Allocate all idle nodes from the coupling map or target as ancilla on the layout.

    A pass for allocating all idle physical qubits (those that exist in coupling
    map or target but not the dag circuit) as ancilla. It will also choose new
    virtual qubits to correspond to those physical ancilla.

    Note:
        This is an analysis pass, and only responsible for choosing physical
        ancilla locations and their corresponding virtual qubits.
        A separate transformation pass must add those virtual qubits to the
        circuit.
    """

    def __init__(self, coupling_map):
        if False:
            print('Hello World!')
        'FullAncillaAllocation initializer.\n\n        Args:\n            coupling_map (Union[CouplingMap, Target]): directed graph representing a coupling map.\n        '
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self.ancilla_name = 'ancilla'

    def run(self, dag):
        if False:
            i = 10
            return i + 15
        'Run the FullAncillaAllocation pass on `dag`.\n\n        Extend the layout with new (physical qubit, virtual qubit) pairs.\n        The dag signals which virtual qubits are already in the circuit.\n        This pass will allocate new virtual qubits such that no collision occurs\n        (i.e. Layout bijectivity is preserved)\n\n        The coupling_map and layout together determine which physical qubits are free.\n\n        Args:\n            dag (DAGCircuit): circuit to analyze\n\n        Returns:\n            DAGCircuit: returns the same dag circuit, unmodified\n\n        Raises:\n            TranspilerError: If there is not layout in the property set or not set at init time.\n        '
        layout = self.property_set.get('layout')
        if layout is None:
            raise TranspilerError('FullAncillaAllocation pass requires property_set["layout"].')
        virtual_bits = layout.get_virtual_bits()
        physical_bits = layout.get_physical_bits()
        if layout:
            FullAncillaAllocation.validate_layout(virtual_bits, set(dag.qubits))
            layout_physical_qubits = list(range(max(physical_bits) + 1))
        else:
            layout_physical_qubits = []
        idle_physical_qubits = [q for q in layout_physical_qubits if q not in physical_bits]
        if self.target:
            idle_physical_qubits = [q for q in range(self.target.num_qubits) if q not in physical_bits]
        elif self.coupling_map:
            idle_physical_qubits = [q for q in self.coupling_map.physical_qubits if q not in physical_bits]
        if idle_physical_qubits:
            if self.ancilla_name in dag.qregs:
                save_prefix = QuantumRegister.prefix
                QuantumRegister.prefix = self.ancilla_name
                qreg = QuantumRegister(len(idle_physical_qubits))
                QuantumRegister.prefix = save_prefix
            else:
                qreg = QuantumRegister(len(idle_physical_qubits), name=self.ancilla_name)
            for (idx, idle_q) in enumerate(idle_physical_qubits):
                self.property_set['layout'][idle_q] = qreg[idx]
            self.property_set['layout'].add_register(qreg)
        return dag

    @staticmethod
    def validate_layout(layout_qubits, dag_qubits):
        if False:
            return 10
        '\n        Checks if all the qregs in ``layout_qregs`` already exist in ``dag_qregs``. Otherwise, raise.\n        '
        for qreg in layout_qubits:
            if qreg not in dag_qubits:
                raise TranspilerError('FullAncillaAllocation: The layout refers to a qubit that does not exist in circuit.')
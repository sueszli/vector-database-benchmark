"""Transform a circuit with virtual qubits into a circuit with physical qubits."""
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout

class ApplyLayout(TransformationPass):
    """Transform a circuit with virtual qubits into a circuit with physical qubits.

    Transforms a DAGCircuit with virtual qubits into a DAGCircuit with physical qubits
    by applying the Layout given in `property_set`.
    Requires either of passes to set/select Layout, e.g. `SetLayout`, `TrivialLayout`.
    Assumes the Layout has full physical qubits.

    If a post layout pass is run and sets the ``post_layout`` property set field with
    a new layout to use after ``ApplyLayout`` has already run once this pass will
    compact the layouts so that we apply
    ``original_virtual`` -> ``existing_layout`` -> ``new_layout`` -> ``new_physical``
    so that the output circuit and layout combination become:
    ``original_virtual`` -> ``new_physical``
    """

    def run(self, dag):
        if False:
            for i in range(10):
                print('nop')
        'Run the ApplyLayout pass on ``dag``.\n\n        Args:\n            dag (DAGCircuit): DAG to map.\n\n        Returns:\n            DAGCircuit: A mapped DAG (with physical qubits).\n\n        Raises:\n            TranspilerError: if no layout is found in ``property_set`` or no full physical qubits.\n        '
        layout = self.property_set['layout']
        if not layout:
            raise TranspilerError("No 'layout' is found in property_set. Please run a Layout pass in advance.")
        if len(layout) != 1 + max(layout.get_physical_bits()):
            raise TranspilerError("The 'layout' must be full (with ancilla).")
        post_layout = self.property_set['post_layout']
        q = QuantumRegister(len(layout), 'q')
        new_dag = DAGCircuit()
        new_dag.add_qreg(q)
        new_dag.metadata = dag.metadata
        new_dag.add_clbits(dag.clbits)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        if post_layout is None:
            self.property_set['original_qubit_indices'] = {bit: index for (index, bit) in enumerate(dag.qubits)}
            for qreg in dag.qregs.values():
                self.property_set['layout'].add_register(qreg)
            virtual_phsyical_map = layout.get_virtual_bits()
            for node in dag.topological_op_nodes():
                qargs = [q[virtual_phsyical_map[qarg]] for qarg in node.qargs]
                new_dag.apply_operation_back(node.op, qargs, node.cargs, check=False)
        else:
            full_layout = Layout()
            old_phys_to_virtual = layout.get_physical_bits()
            new_virtual_to_physical = post_layout.get_virtual_bits()
            phys_map = list(range(len(new_dag.qubits)))
            for (new_virt, new_phys) in new_virtual_to_physical.items():
                old_phys = dag.find_bit(new_virt).index
                old_virt = old_phys_to_virtual[old_phys]
                phys_map[old_phys] = new_phys
                full_layout.add(old_virt, new_phys)
            for reg in layout.get_registers():
                full_layout.add_register(reg)
            for node in dag.topological_op_nodes():
                qargs = [q[new_virtual_to_physical[qarg]] for qarg in node.qargs]
                new_dag.apply_operation_back(node.op, qargs, node.cargs, check=False)
            self.property_set['layout'] = full_layout
            if (final_layout := self.property_set['final_layout']) is not None:
                final_layout_mapping = {new_dag.qubits[phys_map[dag.find_bit(old_virt).index]]: phys_map[old_phys] for (old_virt, old_phys) in final_layout.get_virtual_bits().items()}
                out_layout = Layout(final_layout_mapping)
                self.property_set['final_layout'] = out_layout
        new_dag._global_phase = dag._global_phase
        return new_dag
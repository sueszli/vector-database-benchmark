"""
Base Fake Qobj.
"""
from qiskit.qobj import QasmQobj, QobjExperimentHeader, QobjHeader, QasmQobjInstruction, QasmQobjExperimentConfig, QasmQobjExperiment, QasmQobjConfig
from .fake_qasm_simulator import FakeQasmSimulator

class FakeQobj(QasmQobj):
    """A fake `Qobj` instance."""

    def __init__(self):
        if False:
            while True:
                i = 10
        qobj_id = 'test_id'
        config = QasmQobjConfig(shots=1024, memory_slots=1)
        header = QobjHeader(backend_name=FakeQasmSimulator().name())
        experiments = [QasmQobjExperiment(instructions=[QasmQobjInstruction(name='barrier', qubits=[1])], header=QobjExperimentHeader(), config=QasmQobjExperimentConfig(seed=123456))]
        super().__init__(qobj_id=qobj_id, config=config, experiments=experiments, header=header)
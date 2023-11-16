"""Mock BackendV2 object without run implemented for testing backwards compat"""
import datetime
import numpy as np
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.measure import Measure
from qiskit.circuit.library.standard_gates import CXGate, UGate, ECRGate, RXGate, SXGate, XGate, RZGate
from qiskit.providers.backend import BackendV2, QubitProperties
from qiskit.providers.options import Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.providers.basicaer.qasm_simulator import QasmSimulatorPy

class FakeBackendV2(BackendV2):
    """A mock backend that doesn't implement run() to test compatibility with Terra internals."""

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(None, name='FakeV2', description='A fake BackendV2 example', online_date=datetime.datetime.utcnow(), backend_version='0.0.1')
        self._qubit_properties = [QubitProperties(t1=6.348783e-05, t2=0.00011223246, frequency=5175380000.0), QubitProperties(t1=7.309352e-05, t2=0.00012683382, frequency=5267220000.0)]
        self._target = Target(qubit_properties=self._qubit_properties)
        self._theta = Parameter('theta')
        self._phi = Parameter('phi')
        self._lam = Parameter('lambda')
        rx_props = {(0,): InstructionProperties(duration=5.23e-08, error=0.00038115), (1,): InstructionProperties(duration=4.52e-08, error=0.00032115)}
        self._target.add_instruction(RXGate(self._theta), rx_props)
        rx_30_props = {(0,): InstructionProperties(duration=1.23e-08, error=0.00018115), (1,): InstructionProperties(duration=1.52e-08, error=0.00012115)}
        self._target.add_instruction(RXGate(np.pi / 6), rx_30_props, name='rx_30')
        u_props = {(0,): InstructionProperties(duration=5.23e-08, error=0.00038115), (1,): InstructionProperties(duration=4.52e-08, error=0.00032115)}
        self._target.add_instruction(UGate(self._theta, self._phi, self._lam), u_props)
        cx_props = {(0, 1): InstructionProperties(duration=5.23e-07, error=0.00098115)}
        self._target.add_instruction(CXGate(), cx_props)
        measure_props = {(0,): InstructionProperties(duration=6e-06, error=5e-06), (1,): InstructionProperties(duration=1e-06, error=9e-06)}
        self._target.add_instruction(Measure(), measure_props)
        ecr_props = {(1, 0): InstructionProperties(duration=4.52e-09, error=1.32115e-05)}
        self._target.add_instruction(ECRGate(), ecr_props)
        self.options.set_validator('shots', (1, 4096))

    @property
    def target(self):
        if False:
            for i in range(10):
                print('nop')
        return self._target

    @property
    def max_circuits(self):
        if False:
            i = 10
            return i + 15
        return None

    @classmethod
    def _default_options(cls):
        if False:
            return 10
        return Options(shots=1024)

    def run(self, run_input, **options):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class FakeBackendV2LegacyQubitProps(FakeBackendV2):
    """Fake backend that doesn't use qubit properties via the target."""

    def qubit_properties(self, qubit):
        if False:
            return 10
        if isinstance(qubit, int):
            return self._qubit_properties[qubit]
        return [self._qubit_properties[i] for i in qubit]

class FakeBackend5QV2(BackendV2):
    """A mock backend that doesn't implement run() to test compatibility with Terra internals."""

    def __init__(self, bidirectional=True):
        if False:
            return 10
        super().__init__(None, name='Fake5QV2', description='A fake BackendV2 example', online_date=datetime.datetime.utcnow(), backend_version='0.0.1')
        qubit_properties = [QubitProperties(t1=6.348783e-05, t2=0.00011223246, frequency=5175380000.0), QubitProperties(t1=7.309352e-05, t2=0.00012683382, frequency=5267220000.0), QubitProperties(t1=7.309352e-05, t2=0.00012683382, frequency=5267220000.0), QubitProperties(t1=7.309352e-05, t2=0.00012683382, frequency=5267220000.0), QubitProperties(t1=7.309352e-05, t2=0.00012683382, frequency=5267220000.0)]
        self._target = Target(qubit_properties=qubit_properties)
        self._theta = Parameter('theta')
        self._phi = Parameter('phi')
        self._lam = Parameter('lambda')
        u_props = {(0,): InstructionProperties(duration=5.23e-08, error=0.00038115), (1,): InstructionProperties(duration=4.52e-08, error=0.00032115), (2,): InstructionProperties(duration=5.23e-08, error=0.00038115), (3,): InstructionProperties(duration=4.52e-08, error=0.00032115), (4,): InstructionProperties(duration=4.52e-08, error=0.00032115)}
        self._target.add_instruction(UGate(self._theta, self._phi, self._lam), u_props)
        cx_props = {(0, 1): InstructionProperties(duration=5.23e-07, error=0.00098115), (3, 4): InstructionProperties(duration=5.23e-07, error=0.00098115)}
        if bidirectional:
            cx_props[1, 0] = InstructionProperties(duration=6.23e-07, error=0.00099115)
            cx_props[4, 3] = InstructionProperties(duration=7.23e-07, error=0.00099115)
        self._target.add_instruction(CXGate(), cx_props)
        measure_props = {(0,): InstructionProperties(duration=6e-06, error=5e-06), (1,): InstructionProperties(duration=1e-06, error=9e-06), (2,): InstructionProperties(duration=6e-06, error=5e-06), (3,): InstructionProperties(duration=1e-06, error=9e-06), (4,): InstructionProperties(duration=1e-06, error=9e-06)}
        self._target.add_instruction(Measure(), measure_props)
        ecr_props = {(1, 2): InstructionProperties(duration=4.52e-09, error=1.32115e-05), (2, 3): InstructionProperties(duration=4.52e-09, error=1.32115e-05)}
        if bidirectional:
            ecr_props[2, 1] = InstructionProperties(duration=5.52e-09, error=2.32115e-05)
            ecr_props[3, 2] = InstructionProperties(duration=5.52e-09, error=2.32115e-05)
        self._target.add_instruction(ECRGate(), ecr_props)
        self.options.set_validator('shots', (1, 4096))

    @property
    def target(self):
        if False:
            return 10
        return self._target

    @property
    def max_circuits(self):
        if False:
            print('Hello World!')
        return None

    @classmethod
    def _default_options(cls):
        if False:
            return 10
        return Options(shots=1024)

    def run(self, run_input, **options):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class FakeBackendSimple(BackendV2):
    """A fake simple backend that wraps BasicAer to implement run()."""

    def __init__(self):
        if False:
            return 10
        super().__init__(None, name='FakeSimpleV2', description='A fake simple BackendV2 example', online_date=datetime.datetime.utcnow(), backend_version='0.0.1')
        self._lam = Parameter('lambda')
        self._target = Target(num_qubits=20)
        self._target.add_instruction(SXGate())
        self._target.add_instruction(XGate())
        self._target.add_instruction(RZGate(self._lam))
        self._target.add_instruction(CXGate())
        self._target.add_instruction(Measure())
        self._runner = QasmSimulatorPy()

    @property
    def target(self):
        if False:
            while True:
                i = 10
        return self._target

    @property
    def max_circuits(self):
        if False:
            while True:
                i = 10
        return None

    @classmethod
    def _default_options(cls):
        if False:
            i = 10
            return i + 15
        return QasmSimulatorPy._default_options()

    def run(self, run_input, **options):
        if False:
            i = 10
            return i + 15
        self._runner._options = self._options
        return self._runner.run(run_input, **options)
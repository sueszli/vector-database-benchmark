import itertools
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter

def build_circuit(width, gates):
    if False:
        while True:
            i = 10
    qr = QuantumRegister(width)
    qc = QuantumCircuit(qr)
    while len(qc) < gates:
        for k in range(width):
            qc.h(qr[k])
        for k in range(width - 1):
            qc.cx(qr[k], qr[k + 1])
    return qc

class CircuitConstructionBench:
    params = ([1, 2, 5, 8, 14, 20], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ['width', 'gates']
    timeout = 600

    def setup(self, width, gates):
        if False:
            return 10
        self.empty_circuit = build_circuit(width, 0)
        self.sample_circuit = build_circuit(width, gates)

    def time_circuit_construction(self, width, gates):
        if False:
            for i in range(10):
                print('nop')
        build_circuit(width, gates)

    def time_circuit_extend(self, _, __):
        if False:
            i = 10
            return i + 15
        self.empty_circuit.extend(self.sample_circuit)

    def time_circuit_copy(self, _, __):
        if False:
            return 10
        self.sample_circuit.copy()

def build_parameterized_circuit(width, gates, param_count):
    if False:
        return 10
    params = [Parameter('param-%s' % x) for x in range(param_count)]
    param_iter = itertools.cycle(params)
    qr = QuantumRegister(width)
    qc = QuantumCircuit(qr)
    while len(qc) < gates:
        for k in range(width):
            param = next(param_iter)
            qc.u2(0, param, qr[k])
        for k in range(width - 1):
            param = next(param_iter)
            qc.crx(param, qr[k], qr[k + 1])
    return (qc, params)

class ParameterizedCircuitConstructionBench:
    params = ([20], [8, 128, 2048, 8192, 32768, 131072], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ['width', 'gates', 'number of params']
    timeout = 600

    def setup(self, _, gates, params):
        if False:
            print('Hello World!')
        if params > gates:
            raise NotImplementedError

    def time_build_parameterized_circuit(self, width, gates, params):
        if False:
            print('Hello World!')
        build_parameterized_circuit(width, gates, params)

class ParameterizedCircuitBindBench:
    params = ([20], [8, 128, 2048, 8192, 32768, 131072], [8, 128, 2048, 8192, 32768, 131072])
    param_names = ['width', 'gates', 'number of params']
    timeout = 600

    def setup(self, width, gates, params):
        if False:
            while True:
                i = 10
        if params > gates:
            raise NotImplementedError
        (self.circuit, self.params) = build_parameterized_circuit(width, gates, params)

    def time_bind_params(self, _, __, ___):
        if False:
            i = 10
            return i + 15
        self.circuit.assign_parameters({x: 3.14 for x in self.params})
import copy
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
try:
    from qiskit.compiler import transpile
    TRANSPILER_SEED_KEYWORD = 'seed_transpiler'
except ImportError:
    from qiskit.transpiler import transpile
    TRANSPILER_SEED_KEYWORD = 'seed_mapper'
try:
    from qiskit.quantum_info.random import random_unitary
    HAS_RANDOM_UNITARY = True
except ImportError:
    from qiskit.tools.qi.qi import random_unitary_matrix
    HAS_RANDOM_UNITARY = False

def make_circuit_ring(nq, depth, seed):
    if False:
        for i in range(10):
            print('nop')
    assert int(nq / 2) == nq / 2
    q = QuantumRegister(nq)
    c = ClassicalRegister(nq)
    qc = QuantumCircuit(q, c)
    offset = 1
    decomposer = OneQubitEulerDecomposer()
    for i in range(nq):
        qc.h(q[i])
    for j in range(depth):
        for i in range(int(nq / 2)):
            k = i * 2 + offset + j % 2
            qc.cx(q[k % nq], q[(k + 1) % nq])
        for i in range(nq):
            if HAS_RANDOM_UNITARY:
                u = random_unitary(2, seed).data
            else:
                u = random_unitary_matrix(2)
            angles = decomposer.angles(u)
            qc.u3(angles[0], angles[1], angles[2], q[i])
    qcm = copy.deepcopy(qc)
    for i in range(nq):
        qcm.measure(q[i], c[i])
    return [qc, qcm, nq]

class BenchRandomCircuitHex:
    params = [2 * i for i in range(2, 8)]
    param_names = ['n_qubits']
    version = 3

    def setup(self, n):
        if False:
            return 10
        depth = 2 * n
        self.seed = 0
        self.circuit = make_circuit_ring(n, depth, self.seed)[0]

    def time_ibmq_backend_transpile(self, _):
        if False:
            return 10
        coupling_map = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4], [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10], [11, 3], [11, 10], [11, 12], [12, 2], [13, 1], [13, 12]]
        transpile(self.circuit, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'], coupling_map=coupling_map, **{TRANSPILER_SEED_KEYWORD: self.seed})

    def track_depth_ibmq_backend_transpile(self, _):
        if False:
            while True:
                i = 10
        coupling_map = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4], [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10], [11, 3], [11, 10], [11, 12], [12, 2], [13, 1], [13, 12]]
        return transpile(self.circuit, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'], coupling_map=coupling_map, **{TRANSPILER_SEED_KEYWORD: self.seed}).depth()
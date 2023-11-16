import os
import numpy as np
from qiskit_experiments.library import StandardRB
try:
    from qiskit.compiler import transpile
    TRANSPILER_SEED_KEYWORD = 'seed_transpiler'
except ImportError:
    from qiskit.transpiler import transpile
    TRANSPILER_SEED_KEYWORD = 'seed_mapper'

def build_rb_circuit(qubits, length_vector, num_samples=1, seed=None):
    if False:
        return 10
    '\n    Randomized Benchmarking sequences.\n    '
    if not seed:
        np.random.seed(10)
    else:
        np.random.seed(seed)
    try:
        rb_exp = StandardRB(qubits, lengths=length_vector, num_samples=num_samples, seed=seed)
    except OSError:
        skip_msg = 'Skipping tests because tables are missing'
        raise NotImplementedError(skip_msg)
    return rb_exp.circuits()

class RandomizedBenchmarkingBenchmark:
    params = ([[0], [0, 1]],)
    param_names = ['qubits']
    version = '0.3.0'
    timeout = 600

    def setup(self, qubits):
        if False:
            print('Hello World!')
        length_vector = np.arange(1, 200, 4)
        num_samples = 1
        self.seed = 10
        self.circuits = build_rb_circuit(qubits=qubits, length_vector=length_vector, num_samples=num_samples, seed=self.seed)

    def teardown(self, _):
        if False:
            print('Hello World!')
        os.environ['QISKIT_IN_PARALLEL'] = 'FALSE'

    def time_ibmq_backend_transpile(self, __):
        if False:
            return 10
        coupling_map = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4], [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10], [11, 3], [11, 10], [11, 12], [12, 2], [13, 1], [13, 12]]
        transpile(self.circuits, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'], coupling_map=coupling_map, optimization_level=0, **{TRANSPILER_SEED_KEYWORD: self.seed})

    def time_ibmq_backend_transpile_single_thread(self, __):
        if False:
            print('Hello World!')
        os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'
        coupling_map = [[1, 0], [1, 2], [2, 3], [4, 3], [4, 10], [5, 4], [5, 6], [5, 9], [6, 8], [7, 8], [9, 8], [9, 10], [11, 3], [11, 10], [11, 12], [12, 2], [13, 1], [13, 12]]
        transpile(self.circuits, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'], coupling_map=coupling_map, optimization_level=0, **{TRANSPILER_SEED_KEYWORD: self.seed})
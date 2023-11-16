import os
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.test.mock import FakeToronto

class TranspilerQualitativeBench:
    params = ([0, 1, 2, 3], ['stochastic', 'sabre'], ['dense', 'noise_adaptive', 'sabre'])
    param_names = ['optimization level', 'routing method', 'layout method']
    timeout = 600

    def setup(self, optimization_level, routing_method, layout_method):
        if False:
            return 10
        self.backend = FakeToronto()
        self.qasm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'qasm'))
        self.depth_4gt10_v1_81 = QuantumCircuit.from_qasm_file(os.path.join(self.qasm_path, 'depth_4gt10-v1_81.qasm'))
        self.depth_4mod5_v0_19 = QuantumCircuit.from_qasm_file(os.path.join(self.qasm_path, 'depth_4mod5-v0_19.qasm'))
        self.depth_mod8_10_178 = QuantumCircuit.from_qasm_file(os.path.join(self.qasm_path, 'depth_mod8-10_178.qasm'))
        self.time_cnt3_5_179 = QuantumCircuit.from_qasm_file(os.path.join(self.qasm_path, 'time_cnt3-5_179.qasm'))
        self.time_cnt3_5_180 = QuantumCircuit.from_qasm_file(os.path.join(self.qasm_path, 'time_cnt3-5_180.qasm'))
        self.time_qft_16 = QuantumCircuit.from_qasm_file(os.path.join(self.qasm_path, 'time_qft_16.qasm'))

    def track_depth_transpile_4gt10_v1_81(self, optimization_level, routing_method, layout_method):
        if False:
            i = 10
            return i + 15
        return transpile(self.depth_4gt10_v1_81, self.backend, routing_method=routing_method, layout_method=layout_method, optimization_level=optimization_level, seed_transpiler=0).depth()

    def track_depth_transpile_4mod5_v0_19(self, optimization_level, routing_method, layout_method):
        if False:
            for i in range(10):
                print('nop')
        return transpile(self.depth_4mod5_v0_19, self.backend, routing_method=routing_method, layout_method=layout_method, optimization_level=optimization_level, seed_transpiler=0).depth()

    def track_depth_transpile_mod8_10_178(self, optimization_level, routing_method, layout_method):
        if False:
            print('Hello World!')
        return transpile(self.depth_mod8_10_178, self.backend, routing_method=routing_method, layout_method=layout_method, optimization_level=optimization_level, seed_transpiler=0).depth()

    def time_transpile_time_cnt3_5_179(self, optimization_level, routing_method, layout_method):
        if False:
            print('Hello World!')
        transpile(self.time_cnt3_5_179, self.backend, routing_method=routing_method, layout_method=layout_method, optimization_level=optimization_level, seed_transpiler=0)

    def time_transpile_time_cnt3_5_180(self, optimization_level, routing_method, layout_method):
        if False:
            return 10
        transpile(self.time_cnt3_5_180, self.backend, routing_method=routing_method, layout_method=layout_method, optimization_level=optimization_level, seed_transpiler=0)

    def time_transpile_time_qft_16(self, optimization_level, routing_method, layout_method):
        if False:
            while True:
                i = 10
        transpile(self.time_qft_16, self.backend, routing_method=routing_method, layout_method=layout_method, optimization_level=optimization_level, seed_transpiler=0)
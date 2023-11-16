from qiskit import transpile
from qiskit.transpiler import CouplingMap
from .utils import build_ripple_adder_circuit

class RippleAdderConstruction:
    params = ([10, 50, 100, 200, 500],)
    param_names = ['size']
    version = 1
    timeout = 600

    def time_build_ripple_adder(self, size):
        if False:
            return 10
        build_ripple_adder_circuit(size)

class RippleAdderTranspile:
    params = ([10, 20], [0, 1, 2, 3])
    param_names = ['size', 'level']
    version = 1
    timeout = 600

    def setup(self, size, _):
        if False:
            for i in range(10):
                print('nop')
        edge_len = int((2 * size + 2) ** 0.5) + 1
        self.coupling_map = CouplingMap.from_grid(edge_len, edge_len)
        self.circuit = build_ripple_adder_circuit(size)

    def time_transpile_square_grid_ripple_adder(self, _, level):
        if False:
            while True:
                i = 10
        transpile(self.circuit, coupling_map=self.coupling_map, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'], optimization_level=level, seed_transpiler=20220125)

    def track_depth_transpile_square_grid_ripple_adder(self, _, level):
        if False:
            return 10
        return transpile(self.circuit, coupling_map=self.coupling_map, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'], optimization_level=level, seed_transpiler=20220125).depth()
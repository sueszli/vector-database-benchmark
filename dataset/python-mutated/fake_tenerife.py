"""
Fake Tenerife device (5 qubit).
"""
import os
import json
from qiskit.providers.models import GateConfig, QasmBackendConfiguration, BackendProperties
from qiskit.providers.fake_provider.fake_backend import FakeBackend

class FakeTenerife(FakeBackend):
    """A fake 5 qubit backend."""

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n\n        .. code-block:: text\n\n                1\n              ↙ ↑\n            0 ← 2 ← 3\n                ↑ ↙\n                4\n        '
        cmap = [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]
        configuration = QasmBackendConfiguration(backend_name='fake_tenerife', backend_version='0.0.0', n_qubits=5, basis_gates=['u1', 'u2', 'u3', 'cx', 'id'], simulator=False, local=True, conditional=False, open_pulse=False, memory=False, max_shots=65536, max_experiments=900, gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')], coupling_map=cmap)
        super().__init__(configuration)

    def properties(self):
        if False:
            i = 10
            return i + 15
        'Returns a snapshot of device properties as recorded on 8/30/19.'
        dirname = os.path.dirname(__file__)
        filename = 'props_tenerife.json'
        with open(os.path.join(dirname, filename)) as f_prop:
            props = json.load(f_prop)
        return BackendProperties.from_dict(props)
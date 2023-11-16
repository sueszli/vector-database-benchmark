"""
Fake 1Q device (1 qubit).
"""
import datetime
from qiskit.providers.models.backendproperties import BackendProperties, Gate, Nduv
from .fake_backend import FakeBackend

class Fake1Q(FakeBackend):
    """A fake 1Q backend."""

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        0\n        '
        mock_time = datetime.datetime.now()
        dt = 1.3333
        configuration = BackendProperties(backend_name='fake_1q', backend_version='0.0.0', n_qubits=1, basis_gates=['u1', 'u2', 'u3', 'cx'], simulator=False, local=True, conditional=False, memory=False, max_shots=1024, qubits=[[Nduv(date=mock_time, name='T1', unit='µs', value=71.9500421005539), Nduv(date=mock_time, name='frequency', unit='MHz', value=4919.96800692)]], gates=[Gate(gate='u1', name='u1_0', qubits=[0], parameters=[Nduv(date=mock_time, name='gate_error', unit='', value=1.0), Nduv(date=mock_time, name='gate_length', unit='ns', value=0.0)]), Gate(gate='u3', name='u3_0', qubits=[0], parameters=[Nduv(date=mock_time, name='gate_error', unit='', value=1.0), Nduv(date=mock_time, name='gate_length', unit='ns', value=2 * dt)]), Gate(gate='u3', name='u3_1', qubits=[1], parameters=[Nduv(date=mock_time, name='gate_error', unit='', value=1.0), Nduv(date=mock_time, name='gate_length', unit='ns', value=4 * dt)]), Gate(gate='cx', name='cx0_1', qubits=[0, 1], parameters=[Nduv(date=mock_time, name='gate_error', unit='', value=1.0), Nduv(date=mock_time, name='gate_length', unit='ns', value=22 * dt)])], coupling_map=None, n_registers=1, last_update_date=mock_time, general=[])
        super().__init__(configuration)
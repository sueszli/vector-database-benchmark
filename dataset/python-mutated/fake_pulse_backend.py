"""
Fake backend abstract class for mock backends supporting OpenPulse.
"""
from qiskit.exceptions import QiskitError
from qiskit.providers.models import PulseBackendConfiguration, PulseDefaults
from .fake_qasm_backend import FakeQasmBackend
from .utils.json_decoder import decode_pulse_defaults

class FakePulseBackend(FakeQasmBackend):
    """A fake pulse backend."""
    defs_filename = None

    def defaults(self):
        if False:
            i = 10
            return i + 15
        'Returns a snapshot of device defaults'
        if not self._defaults:
            self._set_defaults_from_json()
        return self._defaults

    def _set_defaults_from_json(self):
        if False:
            while True:
                i = 10
        if not self.props_filename:
            raise QiskitError('No properties file has been defined')
        defs = self._load_json(self.defs_filename)
        decode_pulse_defaults(defs)
        self._defaults = PulseDefaults.from_dict(defs)

    def _get_config_from_dict(self, conf):
        if False:
            for i in range(10):
                print('nop')
        return PulseBackendConfiguration.from_dict(conf)
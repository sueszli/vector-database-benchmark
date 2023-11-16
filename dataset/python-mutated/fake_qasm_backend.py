"""
Fake backend abstract class for mock backends.
"""
import json
import os
from qiskit.exceptions import QiskitError
from qiskit.providers.models import BackendProperties, QasmBackendConfiguration
from .utils.json_decoder import decode_backend_configuration, decode_backend_properties
from .fake_backend import FakeBackend

class FakeQasmBackend(FakeBackend):
    """A fake OpenQASM backend."""
    dirname = None
    conf_filename = None
    props_filename = None
    backend_name = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        configuration = self._get_conf_from_json()
        self._defaults = None
        self._properties = None
        super().__init__(configuration)

    def properties(self):
        if False:
            i = 10
            return i + 15
        'Returns a snapshot of device properties'
        if not self._properties:
            self._set_props_from_json()
        return self._properties

    def _get_conf_from_json(self):
        if False:
            print('Hello World!')
        if not self.conf_filename:
            raise QiskitError('No configuration file has been defined')
        conf = self._load_json(self.conf_filename)
        decode_backend_configuration(conf)
        configuration = self._get_config_from_dict(conf)
        configuration.backend_name = self.backend_name
        return configuration

    def _set_props_from_json(self):
        if False:
            while True:
                i = 10
        if not self.props_filename:
            raise QiskitError('No properties file has been defined')
        props = self._load_json(self.props_filename)
        decode_backend_properties(props)
        self._properties = BackendProperties.from_dict(props)

    def _load_json(self, filename):
        if False:
            i = 10
            return i + 15
        with open(os.path.join(self.dirname, filename)) as f_json:
            the_json = json.load(f_json)
        return the_json

    def _get_config_from_dict(self, conf):
        if False:
            for i in range(10):
                print('nop')
        return QasmBackendConfiguration.from_dict(conf)
"""Provider for Basic Aer simulator backends."""
from collections import OrderedDict
import logging
from qiskit.exceptions import QiskitError
from qiskit.providers.provider import ProviderV1
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.providerutils import resolve_backend_name, filter_backends
from .qasm_simulator import QasmSimulatorPy
from .statevector_simulator import StatevectorSimulatorPy
from .unitary_simulator import UnitarySimulatorPy
logger = logging.getLogger(__name__)
SIMULATORS = [QasmSimulatorPy, StatevectorSimulatorPy, UnitarySimulatorPy]

class BasicAerProvider(ProviderV1):
    """Provider for Basic Aer backends."""

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._backends = self._verify_backends()

    def get_backend(self, name=None, **kwargs):
        if False:
            print('Hello World!')
        backends = self._backends.values()
        if name:
            try:
                resolved_name = resolve_backend_name(name, backends, self._deprecated_backend_names(), {})
                name = resolved_name
            except LookupError as ex:
                raise QiskitBackendNotFoundError(f"The '{name}' backend is not installed in your system.") from ex
        return super().get_backend(name=name, **kwargs)

    def backends(self, name=None, filters=None, **kwargs):
        if False:
            return 10
        backends = self._backends.values()
        if name:
            try:
                resolved_name = resolve_backend_name(name, backends, self._deprecated_backend_names(), {})
                backends = [backend for backend in backends if backend.name() == resolved_name]
            except LookupError:
                return []
        return filter_backends(backends, filters=filters, **kwargs)

    @staticmethod
    def _deprecated_backend_names():
        if False:
            while True:
                i = 10
        'Returns deprecated backend names.'
        return {'qasm_simulator_py': 'qasm_simulator', 'statevector_simulator_py': 'statevector_simulator', 'unitary_simulator_py': 'unitary_simulator', 'local_qasm_simulator_py': 'qasm_simulator', 'local_statevector_simulator_py': 'statevector_simulator', 'local_unitary_simulator_py': 'unitary_simulator', 'local_unitary_simulator': 'unitary_simulator'}

    def _verify_backends(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the Basic Aer backends in `BACKENDS` that are\n        effectively available (as some of them might depend on the presence\n        of an optional dependency or on the existence of a binary).\n\n        Returns:\n            dict[str:Backend]: a dict of Basic Aer backend instances for\n                the backends that could be instantiated, keyed by backend name.\n        '
        ret = OrderedDict()
        for backend_cls in SIMULATORS:
            backend_instance = self._get_backend_instance(backend_cls)
            backend_name = backend_instance.name()
            ret[backend_name] = backend_instance
        return ret

    def _get_backend_instance(self, backend_cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an instance of a backend from its class.\n\n        Args:\n            backend_cls (class): backend class.\n        Returns:\n            Backend: a backend instance.\n        Raises:\n            QiskitError: if the backend could not be instantiated.\n        '
        try:
            backend_instance = backend_cls(provider=self)
        except Exception as err:
            raise QiskitError(f'Backend {backend_cls} could not be instantiated: {err}') from err
        return backend_instance

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'BasicAer'
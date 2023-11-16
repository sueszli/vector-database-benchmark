"""backend utility functions"""
import logging
from qiskit.utils.deprecation import deprecate_func
logger = logging.getLogger(__name__)
_UNSUPPORTED_BACKENDS = ['unitary_simulator', 'clifford_simulator']

class ProviderCheck:
    """Contains Provider verification info."""

    def __init__(self) -> None:
        if False:
            return 10
        self.has_ibmq = False
        self.checked_ibmq = False
        self.has_aer = False
        self.checked_aer = False
_PROVIDER_CHECK = ProviderCheck()

def _get_backend_interface_version(backend):
    if False:
        print('Hello World!')
    'Get the backend version int.'
    backend_interface_version = getattr(backend, 'version', None)
    return backend_interface_version

def _get_backend_provider(backend):
    if False:
        for i in range(10):
            print('nop')
    backend_interface_version = _get_backend_interface_version(backend)
    if backend_interface_version > 1:
        provider = backend.provider
    else:
        provider = backend.provider()
    return provider

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def has_ibmq():
    if False:
        i = 10
        return i + 15
    'Check if IBMQ is installed.'
    if not _PROVIDER_CHECK.checked_ibmq:
        try:
            from qiskit.providers.ibmq import IBMQFactory
            from qiskit.providers.ibmq.accountprovider import AccountProvider
            _PROVIDER_CHECK.has_ibmq = True
        except Exception as ex:
            _PROVIDER_CHECK.has_ibmq = False
            logger.debug("IBMQFactory/AccountProvider not loaded: '%s'", str(ex))
        _PROVIDER_CHECK.checked_ibmq = True
    return _PROVIDER_CHECK.has_ibmq

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def has_aer():
    if False:
        return 10
    'Check if Aer is installed.'
    if not _PROVIDER_CHECK.checked_aer:
        try:
            from qiskit.providers.aer import AerProvider
            _PROVIDER_CHECK.has_aer = True
        except Exception as ex:
            _PROVIDER_CHECK.has_aer = False
            logger.debug("AerProvider not loaded: '%s'", str(ex))
        _PROVIDER_CHECK.checked_aer = True
    return _PROVIDER_CHECK.has_aer

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def is_aer_provider(backend):
    if False:
        for i in range(10):
            print('nop')
    'Detect whether or not backend is from Aer provider.\n\n    Args:\n        backend (Backend): backend instance\n    Returns:\n        bool: True is AerProvider\n    '
    if has_aer():
        from qiskit.providers.aer import AerProvider
        if isinstance(_get_backend_provider(backend), AerProvider):
            return True
        from qiskit.providers.aer.backends.aerbackend import AerBackend
        return isinstance(backend, AerBackend)
    return False

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def is_basicaer_provider(backend):
    if False:
        i = 10
        return i + 15
    'Detect whether or not backend is from BasicAer provider.\n\n    Args:\n        backend (Backend): backend instance\n    Returns:\n        bool: True is BasicAer\n    '
    from qiskit.providers.basicaer import BasicAerProvider
    return isinstance(_get_backend_provider(backend), BasicAerProvider)

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def is_ibmq_provider(backend):
    if False:
        print('Hello World!')
    'Detect whether or not backend is from IBMQ provider.\n\n    Args:\n        backend (Backend): backend instance\n    Returns:\n        bool: True is IBMQ\n    '
    if has_ibmq():
        from qiskit.providers.ibmq.accountprovider import AccountProvider
        return isinstance(_get_backend_provider(backend), AccountProvider)
    return False

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def is_aer_statevector_backend(backend):
    if False:
        while True:
            i = 10
    '\n    Return True if backend object is statevector and from Aer provider.\n\n    Args:\n        backend (Backend): backend instance\n    Returns:\n        bool: True is statevector\n    '
    return is_statevector_backend(backend) and is_aer_provider(backend)

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def is_statevector_backend(backend):
    if False:
        return 10
    '\n    Return True if backend object is statevector.\n\n    Args:\n        backend (Backend): backend instance\n    Returns:\n        bool: True is statevector\n    '
    if backend is None:
        return False
    backend_interface_version = _get_backend_interface_version(backend)
    if has_aer():
        from qiskit.providers.aer.backends import AerSimulator, StatevectorSimulator
        if isinstance(backend, StatevectorSimulator):
            return True
        if isinstance(backend, AerSimulator):
            if backend_interface_version <= 1:
                name = backend.name()
            else:
                name = backend.name
            if 'aer_simulator_statevector' in name:
                return True
    if backend_interface_version <= 1:
        return backend.name().startswith('statevector')
    else:
        return backend.name.startswith('statevector')

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def is_simulator_backend(backend):
    if False:
        print('Hello World!')
    '\n    Return True if backend is a simulator.\n\n    Args:\n        backend (Backend): backend instance\n    Returns:\n        bool: True is a simulator\n    '
    backend_interface_version = _get_backend_interface_version(backend)
    if backend_interface_version <= 1:
        return backend.configuration().simulator
    return False

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def is_local_backend(backend):
    if False:
        return 10
    '\n    Return True if backend is a local backend.\n\n    Args:\n        backend (Backend): backend instance\n    Returns:\n        bool: True is a local backend\n    '
    backend_interface_version = _get_backend_interface_version(backend)
    if backend_interface_version <= 1:
        return backend.configuration().local
    return False

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def is_aer_qasm(backend):
    if False:
        return 10
    '\n    Return True if backend is Aer Qasm simulator\n    Args:\n        backend (Backend): backend instance\n\n    Returns:\n        bool: True is Aer Qasm simulator\n    '
    ret = False
    if is_aer_provider(backend):
        if not is_statevector_backend(backend):
            ret = True
    return ret

@deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
def support_backend_options(backend):
    if False:
        i = 10
        return i + 15
    '\n    Return True if backend supports backend_options\n    Args:\n        backend (Backend): backend instance\n\n    Returns:\n        bool: True is support backend_options\n    '
    ret = False
    if is_basicaer_provider(backend) or is_aer_provider(backend):
        ret = True
    return ret
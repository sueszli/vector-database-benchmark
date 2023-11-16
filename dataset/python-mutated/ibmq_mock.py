"""Mock functions for qiskit.IBMQ."""
from unittest.mock import MagicMock
import qiskit
from qiskit.providers import fake_provider as backend_mocks

def mock_get_backend(backend):
    if False:
        i = 10
        return i + 15
    'Replace qiskit.IBMQ with a mock that returns a single backend.\n\n    Note this will set the value of qiskit.IBMQ to a MagicMock object. It is\n    intended to be run as part of docstrings with jupyter-example in a hidden\n    cell so that later examples which rely on ibmq devices so that the docs can\n    be built without requiring configured credentials. If used outside of this\n    context be aware that you will have to manually restore qiskit.IBMQ the\n    value to qiskit.providers.ibmq.IBMQ after you finish using your mock.\n\n    Args:\n        backend (str): The class name as a string for the fake device to\n            return from the mock IBMQ object. For example, FakeVigo.\n    Raises:\n        NameError: If the specified value of backend\n    '
    mock_ibmq = MagicMock()
    mock_provider = MagicMock()
    if not hasattr(backend_mocks, backend):
        raise NameError('The specified backend name is not a valid mock from qiskit.providers.fake_provider.')
    fake_backend = getattr(backend_mocks, backend)()
    mock_provider.get_backend.return_value = fake_backend
    mock_ibmq.get_provider.return_value = mock_provider
    qiskit.IBMQ = mock_ibmq
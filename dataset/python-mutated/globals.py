"""
A state holder for testing, this is only intended to hold and store testing data to be
used by the test helpers to prune unsupported data.

Should not be used inside any of the test functions.
"""
from dataclasses import dataclass
from .pipeline_helper import get_frontend_config
available_frameworks = ['numpy', 'jax', 'tensorflow', 'torch', 'paddle', 'mxnet', 'scipy']
mod_frontend = {'tensorflow': None, 'numpy': None, 'jax': None, 'torch': None, 'mindspore': None, 'scipy': None, 'paddle': None}
mod_backend = {'numpy': None, 'jax': None, 'tensorflow': None, 'torch': None, 'paddle': None, 'mxnet': None}
_Notsetval = object()
CURRENT_GROUND_TRUTH_BACKEND: callable = _Notsetval
CURRENT_BACKEND: callable = _Notsetval
CURRENT_FRONTEND: callable = _Notsetval
CURRENT_FRONTEND_CONFIG: _Notsetval
CURRENT_RUNNING_TEST = _Notsetval
CURRENT_DEVICE = _Notsetval
CURRENT_DEVICE_STRIPPED = _Notsetval
CURRENT_FRONTEND_STR = None

@dataclass(frozen=True)
class TestData:
    test_fn: callable
    fn_tree: str
    fn_name: str
    supported_device_dtypes: dict = None
    is_method: bool = False

class InterruptedTest(BaseException):
    """Indicate that a test tried to write global attributes while a test is running."""

    def __init__(self, test_interrupted):
        if False:
            print('Hello World!')
        super.__init__(f'{test_interrupted} was interrupted during execution.')

def setup_api_test(backend: str, ground_truth_backend: str, device: str, test_data: TestData=None):
    if False:
        while True:
            i = 10
    if test_data is not None:
        _set_test_data(test_data)
    if ground_truth_backend is not None:
        _set_ground_truth_backend(ground_truth_backend)
    _set_backend(backend)
    _set_device(device)

def teardown_api_test():
    if False:
        while True:
            i = 10
    _unset_test_data()
    _unset_ground_truth_backend()
    _unset_backend()
    _unset_device()

def setup_frontend_test(frontend: str, backend: str, device: str, test_data: TestData):
    if False:
        i = 10
        return i + 15
    if test_data is not None:
        _set_test_data(test_data)
    _set_frontend(frontend)
    _set_backend(backend)
    _set_device(device)

def teardown_frontend_test():
    if False:
        print('Hello World!')
    _unset_test_data()
    _unset_frontend()
    _unset_backend()
    _unset_device()

def _set_test_data(test_data: TestData):
    if False:
        while True:
            i = 10
    global CURRENT_RUNNING_TEST
    if CURRENT_RUNNING_TEST is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_RUNNING_TEST = test_data

def _set_frontend(framework: str):
    if False:
        i = 10
        return i + 15
    global CURRENT_FRONTEND
    global CURRENT_FRONTEND_CONFIG
    if CURRENT_FRONTEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_FRONTEND_CONFIG = get_frontend_config(framework)
    CURRENT_FRONTEND = framework

def _set_backend(framework: str):
    if False:
        while True:
            i = 10
    global CURRENT_BACKEND
    if CURRENT_BACKEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_BACKEND = framework

def _set_ground_truth_backend(framework: str):
    if False:
        while True:
            i = 10
    global CURRENT_GROUND_TRUTH_BACKEND
    if CURRENT_GROUND_TRUTH_BACKEND is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_GROUND_TRUTH_BACKEND = framework

def _set_device(device: str):
    if False:
        return 10
    global CURRENT_DEVICE, CURRENT_DEVICE_STRIPPED
    if CURRENT_DEVICE is not _Notsetval or CURRENT_DEVICE_STRIPPED is not _Notsetval:
        raise InterruptedTest(CURRENT_RUNNING_TEST)
    CURRENT_DEVICE = device
    CURRENT_DEVICE_STRIPPED = device.partition(':')[0]

def _unset_test_data():
    if False:
        while True:
            i = 10
    global CURRENT_RUNNING_TEST
    CURRENT_RUNNING_TEST = _Notsetval

def _unset_frontend():
    if False:
        for i in range(10):
            print('nop')
    global CURRENT_FRONTEND, CURRENT_FRONTEND_CONFIG
    CURRENT_FRONTEND = _Notsetval
    CURRENT_FRONTEND_CONFIG = _Notsetval

def _unset_backend():
    if False:
        for i in range(10):
            print('nop')
    global CURRENT_BACKEND
    CURRENT_BACKEND = _Notsetval

def _unset_ground_truth_backend():
    if False:
        while True:
            i = 10
    global CURRENT_GROUND_TRUTH_BACKEND
    CURRENT_GROUND_TRUTH_BACKEND = _Notsetval

def _unset_device():
    if False:
        while True:
            i = 10
    global CURRENT_DEVICE, CURRENT_DEVICE_STRIPPED
    CURRENT_DEVICE = _Notsetval
    CURRENT_DEVICE_STRIPPED = _Notsetval
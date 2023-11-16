"""Utility to set up DTensor backend in tests."""
import multiprocessing
import os
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.python.platform import test as tf_test

class DTensorTestBackendConfigurator:
    """Configurate test backends."""

    def __init__(self, test_case: tf_test.TestCase):
        if False:
            i = 10
            return i + 15
        self._test_case = test_case

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if accelerator_util.is_initialized():
            accelerator_util.shutdown_accelerator_system()

def config_test_mesh(mesh: layout_lib.Mesh):
    if False:
        i = 10
        return i + 15
    'No Op.\n\n  Args:\n    mesh: The DTensor mesh.\n  '
    if config.backend_is_pw():
        del mesh

def slice_host_devices_for_multiworker(num_clients, client_id, ports):
    if False:
        while True:
            i = 10
    'Configure the current process to only use a slice of devices.'
    if num_clients == 0:
        del os.environ['CUDA_VISIBLE_DEVICES']
        del os.environ['HIP_VISIBLE_DEVICES']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{client_id}'
        os.environ['HIP_VISIBLE_DEVICES'] = f'{client_id}'
        os.environ['CLOUD_TPU_TASK_ID'] = f'{client_id}'
        if 'tpu' in DTENSOR_TEST_UTIL_BACKEND.value:
            del ports
            raise NotImplementedError('OSS multi-client tests of TPU is not supported.')

def get_mp_context():
    if False:
        i = 10
        return i + 15
    return multiprocessing.get_context('forkserver')

def handle_test_main(main, *args, **kwargs):
    if False:
        return 10
    main(*args, **kwargs)
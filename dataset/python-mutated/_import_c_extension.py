import atexit
import logging
import sys
from caffe2.python import extension_loader
with extension_loader.DlopenGuard():
    has_hip_support = False
    has_cuda_support = False
    has_gpu_support = False
    try:
        from caffe2.python.caffe2_pybind11_state_gpu import *
        if num_cuda_devices():
            has_gpu_support = has_cuda_support = True
    except ImportError as gpu_e:
        logging.info('Failed to import cuda module: {}'.format(gpu_e))
        try:
            from caffe2.python.caffe2_pybind11_state_hip import *
            has_gpu_support = has_hip_support = True
            logging.info('This caffe2 python run has AMD GPU support!')
        except ImportError as hip_e:
            logging.info('Failed to import AMD hip module: {}'.format(hip_e))
            logging.warning('This caffe2 python run failed to load cuda module:{},and AMD hip module:{}.Will run in CPU only mode.'.format(gpu_e, hip_e))
            try:
                from caffe2.python.caffe2_pybind11_state import *
            except ImportError as cpu_e:
                logging.critical('Cannot load caffe2.python. Error: {0}'.format(str(cpu_e)))
                sys.exit(1)
atexit.register(on_module_exit)

def _TensorCPU_shape(self):
    if False:
        i = 10
        return i + 15
    return tuple(self._shape)

def _TensorCPU_reshape(self, shape):
    if False:
        return 10
    return self._reshape(list(shape))
TensorCPU.shape = property(_TensorCPU_shape)
TensorCPU.reshape = _TensorCPU_reshape
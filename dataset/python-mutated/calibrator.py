import os
import logging
import tensorrt as trt
import pycuda.driver as cuda
logger = logging.getLogger(__name__)

class Calibrator(trt.IInt8Calibrator):

    def __init__(self, training_data, cache_file, batch_size=64, algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        training_data : numpy array\n            The data using to calibrate quantization model\n        cache_file : str\n            The path user want to store calibrate cache file\n        batch_size : int\n            The batch_size of calibrating process\n        algorithm : tensorrt.tensorrt.CalibrationAlgoType\n            The algorithms of calibrating contains LEGACY_CALIBRATION,\n            ENTROPY_CALIBRATION, ENTROPY_CALIBRATION_2, MINMAX_CALIBRATION.\n            Please refer to https://docs.nvidia.com/deeplearning/tensorrt/api/\n            python_api/infer/Int8/Calibrator.html for detail\n        '
        trt.IInt8Calibrator.__init__(self)
        self.algorithm = algorithm
        self.cache_file = cache_file
        self.data = training_data
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_algorithm(self):
        if False:
            print('Hello World!')
        return self.algorithm

    def get_batch_size(self):
        if False:
            print('Hello World!')
        return self.batch_size

    def get_batch(self, names):
        if False:
            print('Hello World!')
        '\n        This function is used to define the way of feeding calibrating data each batch.\n\n        Parameters\n        ----------\n        names : str\n             The names of the network inputs for each object in the bindings array\n\n        Returns\n        -------\n        list\n            A list of device memory pointers set to the memory containing each network\n            input data, or an empty list if there are no more batches for calibration.\n            You can allocate these device buffers with pycuda, for example, and then\n            cast them to int to retrieve the pointer\n        '
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None
        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            logger.info('Calibrating batch %d, containing %d images', current_batch, self.batch_size)
        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        memory_pointers = [self.device_input]
        return memory_pointers

    def read_calibration_cache(self):
        if False:
            i = 10
            return i + 15
        '\n        If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.\n\n        Returns\n        -------\n        cache object\n            A cache object which contains calibration parameters for quantization\n        '
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        if False:
            i = 10
            return i + 15
        '\n        Write calibration cache to specific path.\n\n        Parameters\n        ----------\n        cache : str\n             The calibration cache to write\n        '
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
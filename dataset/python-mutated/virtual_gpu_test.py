"""Tests for multiple virtual GPU support."""
import random
import numpy as np
from google.protobuf import text_format
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

class VirtualGpuTestUtil(object):

    def __init__(self, dim=1000, num_ops=100, virtual_devices_per_gpu=None, device_probabilities=None):
        if False:
            for i in range(10):
                print('nop')
        self._dim = dim
        self._num_ops = num_ops
        if virtual_devices_per_gpu is None:
            self._virtual_devices_per_gpu = [3]
        else:
            self._virtual_devices_per_gpu = virtual_devices_per_gpu
        self._visible_device_list = [i for i in range(len(self._virtual_devices_per_gpu))]
        gpu_devices = ['/gpu:' + str(i) for i in range(sum(self._virtual_devices_per_gpu))]
        self.devices = ['/cpu:0'] + gpu_devices
        self._num_devices = len(self.devices)
        self._mem_limits_mb = [[1 << 11] * i for i in self._virtual_devices_per_gpu]
        self.config = self._GetSessionConfig()
        if device_probabilities is not None:
            self._device_probabilities = list(device_probabilities)
            for i in range(1, self._num_devices):
                self._device_probabilities[i] += self._device_probabilities[i - 1]
        else:
            step = 1.0 / self._num_devices
            self._device_probabilities = [(x + 1) * step for x in range(self._num_devices)]
        self._device_probabilities[self._num_devices - 1] = 1.1
        logging.info('dim: %d', self._dim)
        logging.info('num_ops: %d', self._num_ops)
        logging.info('visible_device_list: %s', str(self._visible_device_list))
        logging.info('virtual_devices_per_gpu: %s', str(self._virtual_devices_per_gpu))
        logging.info('mem_limits: %s', str(self._mem_limits_mb))
        logging.info('devices: %s', str(self.devices))
        logging.info('config: %s', text_format.MessageToString(self.config))
        logging.info('device_probabilities: %s', str(self._device_probabilities))

    def _GetSessionConfig(self):
        if False:
            while True:
                i = 10
        virtual_device_gpu_options = config_pb2.GPUOptions(visible_device_list=','.join((str(d) for d in self._visible_device_list)), experimental=config_pb2.GPUOptions.Experimental(virtual_devices=[config_pb2.GPUOptions.Experimental.VirtualDevices(memory_limit_mb=i) for i in self._mem_limits_mb]))
        return config_pb2.ConfigProto(gpu_options=virtual_device_gpu_options)

    def _GenerateOperationPlacement(self):
        if False:
            while True:
                i = 10
        result = []
        for unused_i in range(self._num_ops):
            op_device = ()
            for unused_j in range(3):
                random_num = random.random()
                for device_index in range(self._num_devices):
                    if self._device_probabilities[device_index] > random_num:
                        op_device += (device_index,)
                        break
            result.append(op_device)
        return result

    def _LogMatrix(self, mat, dim):
        if False:
            for i in range(10):
                print('nop')
        logging.info('---- printing the first 10*10 submatrix ----')
        for i in range(min(10, dim)):
            row = ''
            for j in range(min(10, dim)):
                row += ' ' + str(mat[i][j])
            logging.info(row)

    def _TestRandomGraphWithDevices(self, sess, seed, op_placement, devices, debug_mode=False):
        if False:
            return 10
        data = []
        shape = (self._dim, self._dim)
        feed_dict = {}
        for i in range(len(devices)):
            with ops.device(devices[i]):
                var = array_ops.placeholder(dtypes.float32, shape=shape)
                np.random.seed(seed + i)
                feed_dict[var] = np.random.uniform(low=0, high=0.1, size=shape).astype(np.float32)
                data.append(var)
        for op in op_placement:
            with ops.device(devices[op[2]]):
                data[op[2]] = math_ops.add(data[op[0]], data[op[1]])
        with ops.device('/cpu:0'):
            s = data[0]
            for i in range(1, len(data)):
                s = math_ops.add(s, data[i])
        if debug_mode:
            logging.info(ops.get_default_graph().as_graph_def())
        result = sess.run(s, feed_dict=feed_dict)
        self._LogMatrix(result, self._dim)
        return result

    def TestRandomGraph(self, sess, op_placement=None, random_seed=None):
        if False:
            while True:
                i = 10
        debug_mode = False
        if op_placement is None:
            op_placement = self._GenerateOperationPlacement()
        else:
            debug_mode = True
        if random_seed is None:
            random_seed = random.randint(0, 1 << 31)
        else:
            debug_mode = True
        logging.info('Virtual gpu functional test for random graph...')
        logging.info('operation placement: %s', str(op_placement))
        logging.info('random seed: %d', random_seed)
        result_vgd = self._TestRandomGraphWithDevices(sess, random_seed, op_placement, self.devices, debug_mode=debug_mode)
        result_cpu = self._TestRandomGraphWithDevices(sess, random_seed, op_placement, ['/cpu:0'] * self._num_devices, debug_mode=debug_mode)
        for i in range(self._dim):
            for j in range(self._dim):
                if result_vgd[i][j] != result_cpu[i][j]:
                    logging.error('Result mismatch at row %d column %d: expected %f, actual %f', i, j, result_cpu[i][j], result_vgd[i][j])
                    logging.error('Devices: %s', self.devices)
                    logging.error('Memory limits (in MB): %s', self._mem_limits_mb)
                    return False
        return True

class VirtualGpuTest(test_util.TensorFlowTestCase):

    def __init__(self, method_name):
        if False:
            return 10
        super(VirtualGpuTest, self).__init__(method_name)
        self._util = VirtualGpuTestUtil()

    @test_util.deprecated_graph_mode_only
    def testStatsContainAllDeviceNames(self):
        if False:
            print('Hello World!')
        with self.session(config=self._util.config) as sess:
            if not test.is_gpu_available(cuda_only=True):
                self.skipTest('No GPU available')
            run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()
            mat_shape = [10, 10]
            data = []
            for d in self._util.devices:
                with ops.device(d):
                    var = variables.Variable(random_ops.random_uniform(mat_shape))
                    self.evaluate(var.initializer)
                    data.append(var)
            s = data[0]
            for i in range(1, len(data)):
                s = math_ops.add(s, data[i])
            sess.run(s, options=run_options, run_metadata=run_metadata)
        self.assertTrue(run_metadata.HasField('step_stats'))
        step_stats = run_metadata.step_stats
        devices = [d.device for d in step_stats.dev_stats]
        self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:0' in devices)
        self.assertTrue('/job:localhost/replica:0/task:0/device:GPU:0' in devices)
        self.assertTrue('/job:localhost/replica:0/task:0/device:GPU:1' in devices)
        self.assertTrue('/job:localhost/replica:0/task:0/device:GPU:2' in devices)

    @test_util.deprecated_graph_mode_only
    def testLargeRandomGraph(self):
        if False:
            print('Hello World!')
        with self.session(config=self._util.config) as sess:
            if not test.is_gpu_available(cuda_only=True):
                self.skipTest('No GPU available')
            for _ in range(5):
                if not self._util.TestRandomGraph(sess):
                    return
if __name__ == '__main__':
    test.main()
import os
import unittest
import numpy as np
from op_test import OpTest, paddle_static_guard
import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

def class_center_sample_numpy(label, classes_list, num_samples):
    if False:
        while True:
            i = 10
    unique_label = np.unique(label)
    nranks = len(classes_list)
    class_interval = np.cumsum(np.insert(classes_list, 0, 0))
    pos_class_center_per_device = []
    unique_label_per_device = []
    for i in range(nranks):
        index = np.logical_and(unique_label >= class_interval[i], unique_label < class_interval[i + 1])
        pos_class_center_per_device.append(unique_label[index] - class_interval[i])
        unique_label_per_device.append(unique_label[index])
    num_samples_per_device = []
    for pos_class_center in pos_class_center_per_device:
        num_samples_per_device.append(max(len(pos_class_center), num_samples))
    sampled_class_interval = np.cumsum(np.insert(num_samples_per_device, 0, 0))
    remapped_dict = {}
    for i in range(nranks):
        for (idx, v) in enumerate(unique_label_per_device[i], sampled_class_interval[i]):
            remapped_dict[v] = idx
    remapped_label = []
    for l in label:
        remapped_label.append(remapped_dict[l])
    return (np.array(remapped_label), np.array(pos_class_center_per_device))

def python_api(label, num_classes=1, num_samples=1, ring_id=0, rank=0, nranks=0, fix_seed=False, seed=0):
    if False:
        while True:
            i = 10
    return paddle.nn.functional.class_center_sample(label, num_classes=num_classes, num_samples=num_samples, group=None)

class TestClassCenterSampleOp(OpTest):

    def initParams(self):
        if False:
            return 10
        self.op_type = 'class_center_sample'
        self.python_api = python_api
        self.batch_size = 20
        self.num_samples = 6
        self.num_classes = 10
        self.seed = 2021

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int64

    def init_fix_seed(self):
        if False:
            return 10
        self.fix_seed = True

    def with_new_comm(self):
        if False:
            return 10
        os.environ['FLAGS_dynamic_static_unified_comm'] = '0'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.with_new_comm()
        self.initParams()
        self.init_dtype()
        self.init_fix_seed()
        label = np.random.randint(0, self.num_classes, (self.batch_size,), dtype=self.dtype)
        (remapped_label, sampled_class_center) = class_center_sample_numpy(label, [self.num_classes], self.num_samples)
        self.inputs = {'Label': label}
        self.outputs = {'RemappedLabel': remapped_label.astype(self.dtype), 'SampledLocalClassCenter': sampled_class_center.astype(self.dtype)}
        self.attrs = {'num_classes': self.num_classes, 'num_samples': self.num_samples, 'seed': self.seed, 'fix_seed': self.fix_seed}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(no_check_set=['SampledLocalClassCenter'], check_pir=True)

class TestClassCenterSampleOpINT32(TestClassCenterSampleOp):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int32

class TestClassCenterSampleOpFixSeed(TestClassCenterSampleOp):

    def init_fix_seed(self):
        if False:
            return 10
        self.fix_seed = True

class TestClassCenterSampleOpWithNewComm(TestClassCenterSampleOp):

    def with_new_comm(self):
        if False:
            print('Hello World!')
        os.environ['FLAGS_dynamic_static_unified_comm'] = '1'

class TestClassCenterSampleV2(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.initParams()
        np.random.seed(self.seed)
        paddle.framework.random._manual_program_seed(2021)
        self.places = [paddle.base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.base.CUDAPlace(0))

    def initParams(self):
        if False:
            print('Hello World!')
        self.batch_size = 10
        self.num_samples = 6
        self.num_classes = 20
        self.seed = 0
        self.init_dtype()

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int64

    def test_static(self):
        if False:
            while True:
                i = 10
        with paddle_static_guard():
            for place in self.places:
                self.check_static_result(place=place)

    @test_with_pir_api
    def check_static_result(self, place):
        if False:
            for i in range(10):
                print('nop')
        with paddle_static_guard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                label_np = np.random.randint(0, self.num_classes, (self.batch_size,), dtype=self.dtype)
                label = paddle.static.data(name='label', shape=[self.batch_size], dtype=self.dtype)
                (remapped_label, sampled_class_index) = paddle.nn.functional.class_center_sample(label, self.num_classes, self.num_samples)
                (remapped_label_np, sampled_class_center_np) = class_center_sample_numpy(label_np, [self.num_classes], self.num_samples)
                exe = paddle.base.Executor(place)
                [remapped_label_res, sampled_class_index_res] = exe.run(feed={'label': label_np}, fetch_list=[remapped_label, sampled_class_index])
                np.testing.assert_allclose(remapped_label_res, remapped_label_np)
                np.testing.assert_allclose(sampled_class_index_res[:len(sampled_class_center_np[0])], sampled_class_center_np[0])

    def test_dynamic(self):
        if False:
            while True:
                i = 10
        for place in self.places:
            self.check_dynamic_result(place=place)

    def check_dynamic_result(self, place):
        if False:
            i = 10
            return i + 15
        with paddle.base.dygraph.guard(place):
            label_np = np.random.randint(0, self.num_classes, (self.batch_size,), dtype=self.dtype)
            label = paddle.to_tensor(label_np, dtype=self.dtype)
            (remapped_label, sampled_class_index) = paddle.nn.functional.class_center_sample(label, self.num_classes, self.num_samples)
            (remapped_label_np, sampled_class_center_np) = class_center_sample_numpy(label_np, [self.num_classes], self.num_samples)
            remapped_label_res = remapped_label.numpy()
            sampled_class_index_res = sampled_class_index.numpy()
            np.testing.assert_allclose(remapped_label_res, remapped_label_np)
            np.testing.assert_allclose(sampled_class_index_res[:len(sampled_class_center_np[0])], sampled_class_center_np[0])

class TestClassCenterSampleV2INT32(TestClassCenterSampleV2):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.int32

class TestClassCenterSampleAPIError(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.initParams()
        np.random.seed(self.seed)
        self.places = [paddle.base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.base.CUDAPlace(0))

    def initParams(self):
        if False:
            for i in range(10):
                print('nop')
        self.batch_size = 20
        self.num_samples = 15
        self.num_classes = 10
        self.seed = 2021
        self.init_dtype()

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.int64

    def test_dynamic_errors(self):
        if False:
            for i in range(10):
                print('nop')

        def test_num_samples():
            if False:
                return 10
            for place in self.places:
                with paddle.base.dygraph.guard(place):
                    label_np = np.random.randint(0, self.num_classes, (self.batch_size,), dtype=self.dtype)
                    label = paddle.to_tensor(label_np)
                    (remapped_label, sampled_class_index) = paddle.nn.functional.class_center_sample(label, self.num_classes, self.num_samples)
        self.assertRaises(ValueError, test_num_samples)

class TestClassCenterSampleAPIError1(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.initParams()
        np.random.seed(self.seed)
        self.places = [paddle.base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.base.CUDAPlace(0))

    def initParams(self):
        if False:
            return 10
        self.batch_size = 5
        self.num_samples = 5
        self.num_classes = 10
        self.seed = 2021
        self.init_dtype()

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.int64

    def test_dynamic_errors(self):
        if False:
            print('Hello World!')

        def test_empty_label():
            if False:
                for i in range(10):
                    print('nop')
            for place in self.places:
                with paddle.base.dygraph.guard(place):
                    label = paddle.to_tensor(np.array([], dtype=self.dtype))
                    (remapped_label, sampled_class_index) = paddle.nn.functional.class_center_sample(label, self.num_classes, self.num_samples)

        def test_group_value():
            if False:
                print('Hello World!')
            for place in self.places:
                with paddle.base.dygraph.guard(place):
                    label_np = np.random.randint(0, self.num_classes, (self.batch_size,), dtype=self.dtype)
                    label = paddle.to_tensor(label_np)
                    (remapped_label, sampled_class_index) = paddle.nn.functional.class_center_sample(label, self.num_classes, self.num_samples, group=True)
        self.assertRaises(ValueError, test_empty_label)
        self.assertRaises(ValueError, test_group_value)
if __name__ == '__main__':
    unittest.main()
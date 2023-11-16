import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16, paddle_static_guard
import paddle
from paddle.base import Program, core, program_guard

def stable_softmax_comm(x):
    if False:
        i = 10
        return i + 15
    shiftx = x - np.max(x)
    deno = np.log(np.sum(np.exp(shiftx)))
    comm = shiftx - deno
    return comm

def margin_cross_entropy(logits, label, axis, margin1, margin2, margin3, scale, reduction=None):
    if False:
        print('Hello World!')
    one_hot_label = np.zeros_like(logits, dtype=logits.dtype)
    for (i, lb) in enumerate(label):
        one_hot_label[i, lb] = 1.0
    theta = np.arccos(logits)
    if margin1 != 1.0:
        theta = margin1 * theta
    if margin2 != 0.0:
        theta = theta + margin2
    margin_cos = np.cos(theta)
    if margin3 != 0.0:
        margin_cos = margin_cos - margin3
    diff = one_hot_label * (margin_cos - logits)
    arc_logits = (logits + diff) * scale
    comm = np.apply_along_axis(stable_softmax_comm, axis, arc_logits)
    loss = (-one_hot_label * comm).sum(axis=axis, keepdims=True)
    softmax = np.exp(comm)
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    return (loss, softmax)

def python_api(logits, label, return_softmax=False, ring_id=0, rank=0, nrank=0, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0):
    if False:
        for i in range(10):
            print('nop')
    return paddle.nn.functional.margin_cross_entropy(logits, label, return_softmax=return_softmax, margin1=margin1, margin2=margin2, margin3=margin3, scale=scale, group=None, reduction=None)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestMarginCrossEntropyOp(OpTest):

    def initParams(self):
        if False:
            for i in range(10):
                print('nop')
        self.python_api = python_api
        self.op_type = 'margin_cross_entropy'
        self.python_out_sig = ['Loss']
        self.axis = -1
        self.batch_dim = 5
        self.feat_dim = 41
        self.num_class = 37

    def init_loss_params(self):
        if False:
            while True:
                i = 10
        self.margin1 = 1.0
        self.margin2 = 0.5
        self.margin3 = 0.0
        self.scale = 2.0

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float64

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.initParams()
        self.init_loss_params()
        self.init_dtype()
        datas = np.random.uniform(-0.99, 0.99, [self.batch_dim, self.feat_dim]).astype(self.dtype)
        datas = datas / np.sqrt(np.sum(np.square(datas), axis=1, keepdims=True))
        weights = np.random.uniform(-0.99, 0.99, [self.feat_dim, self.num_class]).astype(self.dtype)
        weights = weights / np.sqrt(np.sum(np.square(weights), axis=0, keepdims=True))
        logits = np.matmul(datas, weights)
        labels = np.random.randint(0, self.num_class, (self.batch_dim,), dtype='int64')
        (loss, softmax) = margin_cross_entropy(logits, labels, self.axis, self.margin1, self.margin2, self.margin3, self.scale)
        self.inputs = {'Logits': logits, 'Label': labels}
        self.outputs = {'Softmax': softmax.astype(self.dtype), 'Loss': loss.astype(self.dtype)}
        self.attrs = {'margin1': self.margin1, 'margin2': self.margin2, 'margin3': self.margin3, 'scale': self.scale}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(core.CUDAPlace(0), atol=1e-05)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(core.CUDAPlace(0), ['Logits'], 'Loss')

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestMarginCrossEntropyOpFP32(TestMarginCrossEntropyOp):

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float32

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad_with_place(core.CUDAPlace(0), ['Logits'], 'Loss', numeric_grad_delta=0.05, max_relative_error=0.05)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestMarginCrossEntropyOpFP16(TestMarginCrossEntropyOp):

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float16

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(core.CUDAPlace(0), atol=0.05)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad_with_place(core.CUDAPlace(0), ['Logits'], 'Loss', numeric_grad_delta=0.6, max_relative_error=0.6)

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA or not support bfloat16')
class TestMarginCrossEntropyBF16Op(OpTest):

    def initParams(self):
        if False:
            for i in range(10):
                print('nop')
        self.python_api = python_api
        self.op_type = 'margin_cross_entropy'
        self.python_out_sig = ['Loss']
        self.axis = -1
        self.batch_dim = 5
        self.feat_dim = 41
        self.num_class = 37

    def init_loss_params(self):
        if False:
            return 10
        self.margin1 = 1.0
        self.margin2 = 0.5
        self.margin3 = 0.0
        self.scale = 2.0

    def init_dtype(self):
        if False:
            return 10
        self.dtype = np.uint16
        self.np_dtype = 'float32'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.initParams()
        self.init_loss_params()
        self.init_dtype()
        datas = np.random.uniform(-0.99, 0.99, [self.batch_dim, self.feat_dim]).astype(self.np_dtype)
        datas = datas / np.sqrt(np.sum(np.square(datas), axis=1, keepdims=True))
        weights = np.random.uniform(-0.99, 0.99, [self.feat_dim, self.num_class]).astype(self.np_dtype)
        weights = weights / np.sqrt(np.sum(np.square(weights), axis=0, keepdims=True))
        logits = np.matmul(datas, weights)
        labels = np.random.randint(0, self.num_class, (self.batch_dim,), dtype='int64')
        (loss, softmax) = margin_cross_entropy(logits, labels, self.axis, self.margin1, self.margin2, self.margin3, self.scale)
        self.inputs = {'Logits': convert_float_to_uint16(logits), 'Label': labels}
        self.outputs = {'Softmax': convert_float_to_uint16(softmax.astype(self.np_dtype)), 'Loss': convert_float_to_uint16(loss.astype(self.np_dtype))}
        self.attrs = {'margin1': self.margin1, 'margin2': self.margin2, 'margin3': self.margin3, 'scale': self.scale}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(core.CUDAPlace(0), atol=0.05)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(core.CUDAPlace(0), ['Logits'], 'Loss', numeric_grad_delta=0.6, max_relative_error=0.6)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestMarginCrossEntropyOpCosFace(TestMarginCrossEntropyOp):

    def init_loss_params(self):
        if False:
            while True:
                i = 10
        self.margin1 = 1.0
        self.margin2 = 0.0
        self.margin3 = 0.35
        self.scale = 2.0

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestMarginCrossEntropyOpSphereFace(TestMarginCrossEntropyOp):

    def init_loss_params(self):
        if False:
            print('Hello World!')
        self.margin1 = 1.35
        self.margin2 = 0.0
        self.margin3 = 0.0
        self.scale = 2.0

class TestMarginCrossEntropyOpCPU(TestMarginCrossEntropyOp):

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.check_output_with_place(core.CPUPlace(), atol=1e-05)
        except RuntimeError:
            pass

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        try:
            self.check_grad_with_place(core.CPUPlace(), ['Logits'], 'Loss')
        except RuntimeError:
            pass

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestMarginCrossEntropyOpV2(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.initParams()
        np.random.seed(self.seed)
        paddle.framework.random._manual_program_seed(self.seed)
        self.places = []
        if core.is_compiled_with_cuda():
            self.places.append(paddle.base.CUDAPlace(0))

    def initParams(self):
        if False:
            return 10
        self.python_out_sig = ['Loss']
        self.seed = 2021
        self.axis = -1
        self.batch_dim = 5
        self.feat_dim = 41
        self.num_class = 37
        self.init_loss_params()
        self.init_dtype()
        self.init_reduction()

    def init_loss_params(self):
        if False:
            return 10
        self.margin1 = 1.0
        self.margin2 = 0.5
        self.margin3 = 0.0
        self.scale = 2.0

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float64

    def init_reduction(self):
        if False:
            while True:
                i = 10
        self.reduction = None

    def test_static(self):
        if False:
            print('Hello World!')
        for place in self.places:
            self.check_static_result(place=place)

    def check_static_result(self, place):
        if False:
            i = 10
            return i + 15
        with paddle_static_guard():
            with program_guard(Program(), Program()):
                datas = np.random.uniform(-0.99, 0.99, [self.batch_dim, self.feat_dim]).astype(self.dtype)
                datas = datas / np.sqrt(np.sum(np.square(datas), axis=1, keepdims=True))
                weights = np.random.uniform(-0.99, 0.99, [self.feat_dim, self.num_class]).astype(self.dtype)
                weights = weights / np.sqrt(np.sum(np.square(weights), axis=0, keepdims=True))
                logits_np = np.matmul(datas, weights)
                labels_np = np.random.randint(0, self.num_class, (self.batch_dim,), dtype='int64')
                (loss_np, softmax_np) = margin_cross_entropy(logits_np, labels_np, self.axis, self.margin1, self.margin2, self.margin3, self.scale, self.reduction)
                logits = paddle.static.data(name='logits', shape=[self.batch_dim, self.num_class], dtype=self.dtype)
                label = paddle.static.data(name='label', shape=[self.batch_dim], dtype='int64')
                (loss, softmax) = paddle.nn.functional.margin_cross_entropy(logits, label, margin1=self.margin1, margin2=self.margin2, margin3=self.margin3, scale=self.scale, return_softmax=True, reduction=self.reduction)
                exe = paddle.base.Executor(place)
                [loss_res, softmax_res] = exe.run(paddle.base.default_main_program(), feed={'logits': logits_np, 'label': labels_np}, fetch_list=[loss, softmax])
                np.testing.assert_allclose(loss_res, loss_np)
                np.testing.assert_allclose(softmax_res, softmax_np)

    def test_dynamic(self):
        if False:
            return 10
        for place in self.places:
            self.check_dynamic_result(place=place)

    def check_dynamic_result(self, place):
        if False:
            while True:
                i = 10
        with paddle.base.dygraph.guard(place):
            datas = np.random.uniform(-0.99, 0.99, [self.batch_dim, self.feat_dim]).astype(self.dtype)
            datas = datas / np.sqrt(np.sum(np.square(datas), axis=1, keepdims=True))
            weights = np.random.uniform(-0.99, 0.99, [self.feat_dim, self.num_class]).astype(self.dtype)
            weights = weights / np.sqrt(np.sum(np.square(weights), axis=0, keepdims=True))
            logits_np = np.matmul(datas, weights)
            labels_np = np.random.randint(0, self.num_class, (self.batch_dim,), dtype='int64')
            (loss_np, softmax_np) = margin_cross_entropy(logits_np, labels_np, self.axis, self.margin1, self.margin2, self.margin3, self.scale, self.reduction)
            logits = paddle.to_tensor(logits_np, dtype=self.dtype)
            labels = paddle.to_tensor(labels_np, dtype='int64')
            (loss, softmax) = paddle.nn.functional.margin_cross_entropy(logits, labels, margin1=self.margin1, margin2=self.margin2, margin3=self.margin3, scale=self.scale, return_softmax=True, reduction=self.reduction)
            loss_res = loss.numpy()
            softmax_res = softmax.numpy()
            np.testing.assert_allclose(loss_res, loss_np)
            np.testing.assert_allclose(softmax_res, softmax_np)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestMarginCrossEntropyOpV3(TestMarginCrossEntropyOpV2):

    def init_reduction(self):
        if False:
            return 10
        self.reduction = 'mean'

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestMarginCrossEntropyOpV4(TestMarginCrossEntropyOpV2):

    def init_reduction(self):
        if False:
            print('Hello World!')
        self.reduction = 'sum'

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestMarginCrossEntropyOpAPIError(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.initParams()
        np.random.seed(self.seed)
        paddle.framework.random._manual_program_seed(self.seed)
        self.places = []
        if core.is_compiled_with_cuda():
            self.places.append(paddle.base.CUDAPlace(0))

    def initParams(self):
        if False:
            return 10
        self.python_api = python_api
        self.python_out_sig = ['Loss']
        self.seed = 2021
        self.axis = -1
        self.batch_dim = 10
        self.feat_dim = 41
        self.num_class = 37
        self.init_loss_params()
        self.init_dtype()

    def init_loss_params(self):
        if False:
            i = 10
            return i + 15
        self.margin1 = 1.0
        self.margin2 = 0.5
        self.margin3 = 0.0
        self.scale = 2.0

    def init_dtype(self):
        if False:
            return 10
        self.dtype = np.float64

    def test_dynamic_errors(self):
        if False:
            print('Hello World!')

        def test_dim():
            if False:
                i = 10
                return i + 15
            for place in self.places:
                with paddle.base.dygraph.guard(place):
                    labels_np = np.random.randint(0, self.num_class, (self.batch_dim, 2), dtype='int64')
                    logits_np = np.random.uniform(-0.99, 0.99, [self.batch_dim, self.num_class]).astype(self.dtype)
                    labels = paddle.to_tensor(labels_np)
                    logits = paddle.to_tensor(logits_np)
                    (loss, softmax) = paddle.nn.functional.margin_cross_entropy(logits, labels, margin1=self.margin1, margin2=self.margin2, margin3=self.margin3, scale=self.scale, return_softmax=True, reduction=None)

        def test_label_type():
            if False:
                print('Hello World!')
            for place in self.places:
                with paddle.base.dygraph.guard(place):
                    labels_np = np.random.uniform(0, self.num_class, (self.batch_dim, 1)).astype(self.dtype)
                    logits_np = np.random.uniform(-0.99, 0.99, [self.batch_dim, self.num_class]).astype(self.dtype)
                    labels = paddle.to_tensor(labels_np)
                    logits = paddle.to_tensor(logits_np)
                    (loss, softmax) = paddle.nn.functional.margin_cross_entropy(logits, labels, margin1=self.margin1, margin2=self.margin2, margin3=self.margin3, scale=self.scale, return_softmax=True, reduction=None)

        def test_group_value():
            if False:
                return 10
            for place in self.places:
                with paddle.base.dygraph.guard(place):
                    labels_np = np.random.randint(0, self.num_class, (self.batch_dim,), dtype='int64')
                    logits_np = np.random.uniform(-0.99, 0.99, [self.batch_dim, self.num_class]).astype(self.dtype)
                    labels = paddle.to_tensor(labels_np)
                    logits = paddle.to_tensor(logits_np)
                    (loss, softmax) = paddle.nn.functional.margin_cross_entropy(logits, labels, margin1=self.margin1, margin2=self.margin2, margin3=self.margin3, scale=self.scale, return_softmax=True, reduction=None, group=True)
        self.assertRaises(ValueError, test_dim)
        self.assertRaises(NotImplementedError, test_label_type)
        self.assertRaises(ValueError, test_group_value)
if __name__ == '__main__':
    unittest.main()
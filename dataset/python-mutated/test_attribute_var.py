import os
import tempfile
import unittest
import numpy as np
import paddle
import paddle.inference as paddle_infer
from paddle.base.framework import OpProtoHolder, Program, program_guard
paddle.enable_static()

class UnittestBase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.temp_dir = tempfile.TemporaryDirectory()
        self.init_info()

    def tearDwon(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def init_info(self):
        if False:
            while True:
                i = 10
        self.shapes = None
        self.save_path = None

    def path_prefix(self):
        if False:
            return 10
        return type(self).__name__

    def infer_prog(self):
        if False:
            for i in range(10):
                print('nop')
        config = paddle_infer.Config(self.save_path + '.pdmodel', self.save_path + '.pdiparams')
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        for (i, shape) in enumerate(self.shapes):
            input_handle = predictor.get_input_handle(input_names[i])
            self.fake_input = np.random.randn(*shape).astype('float32')
            input_handle.reshape(shape)
            input_handle.copy_from_cpu(self.fake_input)
        predictor.run()
        output_names = predictor.get_output_names()
        res = []
        for out_name in output_names:
            output_handle = predictor.get_output_handle(out_name)
            output_data = output_handle.copy_to_cpu()
            res.append(output_data)
        if len(output_names) == 1:
            res = res[0]
        return res

class TestDropout(UnittestBase):

    def init_info(self):
        if False:
            while True:
                i = 10
        self.shapes = [[10, 10]]
        self.save_path = os.path.join(self.temp_dir.name, 'dropout')

    def test_static(self):
        if False:
            for i in range(10):
                print('nop')
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(10, 10)
            x = paddle.randn(self.shapes[0])
            x.stop_gradient = False
            feat = fc(x)
            p = paddle.randn([1])
            out = paddle.nn.functional.dropout(feat, p=p)
            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            self.assertTrue('Var[' in str(main_prog))
            exe = paddle.static.Executor()
            exe.run(starup_prog)
            res = exe.run(fetch_list=[x, out])
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            infer_out = self.infer_prog()
            self.assertEqual(infer_out.shape, (10, 10))
            self.assertEqual(main_prog.block(0).ops[4].all_attrs()['dropout_prob'].name, p.name)

class TestTileTensorList(UnittestBase):

    def init_info(self):
        if False:
            for i in range(10):
                print('nop')
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, 'tile_tensors')

    def _test_static(self):
        if False:
            for i in range(10):
                print('nop')
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)
            shape0 = paddle.full([1], 1, dtype='int32')
            shape1 = paddle.full([1], 2, dtype='int32')
            shape = [3, shape1, shape0]
            out = paddle.tile(feat, shape)
            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            self.assertTrue('Vars[' in str(main_prog))
            exe = paddle.static.Executor()
            exe.run(starup_prog)
            res = exe.run(fetch_list=[x, out])
            self.assertEqual(res[1].shape, (6, 6, 10))
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            infer_out = self.infer_prog()
            self.assertEqual(infer_out.shape, (6, 6, 10))

class TestTileTensor(UnittestBase):

    def init_info(self):
        if False:
            while True:
                i = 10
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, 'tile_tensor')

    def _test_static(self):
        if False:
            while True:
                i = 10
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)
            shape = paddle.assign([3, 2, 1])
            out = paddle.tile(feat, shape)
            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            self.assertTrue('Var[' in str(main_prog))
            exe = paddle.static.Executor()
            exe.run(starup_prog)
            res = exe.run(fetch_list=[x, out])
            self.assertEqual(res[1].shape, (6, 6, 10))
            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            infer_out = self.infer_prog()
            self.assertEqual(infer_out.shape, (6, 6, 10))

class TestRegiterSupportTensorInOpMaker(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.all_protos = OpProtoHolder.instance()
        self.support_tensor_attrs = {'dropout': ['dropout_prob'], 'tile': ['repeat_times']}
        self.not_support_tensor_attrs = {'svd': ['full_matrices']}

    def test_support_tensor(self):
        if False:
            while True:
                i = 10
        for (op_type, attr_names) in self.support_tensor_attrs.items():
            for attr_name in attr_names:
                self.assertTrue(self.is_support_tensor_attr(op_type, attr_name))
        for (op_type, attr_names) in self.not_support_tensor_attrs.items():
            for attr_name in attr_names:
                self.assertFalse(self.is_support_tensor_attr(op_type, attr_name))

    def is_support_tensor_attr(self, op_type, attr_name):
        if False:
            i = 10
            return i + 15
        proto = self.all_protos.get_op_proto(op_type)
        for attr in proto.attrs:
            if attr.name == attr_name:
                return attr.support_tensor
        raise RuntimeError('Not found attribute : ', attr_name)
if __name__ == '__main__':
    unittest.main()
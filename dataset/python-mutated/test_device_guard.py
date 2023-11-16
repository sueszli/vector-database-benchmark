import unittest
import warnings
import paddle
from paddle.base import core
paddle.enable_static()

def execute(main_program, startup_program):
    if False:
        i = 10
        return i + 15
    if paddle.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(startup_program)
    exe.run(main_program)

def get_vaild_warning_num(warning, w):
    if False:
        for i in range(10):
            print('nop')
    num = 0
    for i in range(len(w)):
        if warning in str(w[i].message):
            num += 1
    return num

class TestDeviceGuard(unittest.TestCase):

    def test_device_guard(self):
        if False:
            while True:
                i = 10
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data1 = paddle.full(shape=[1, 3, 8, 8], fill_value=0.5, dtype='float32')
            data2 = paddle.full(shape=[1, 3, 5, 5], fill_value=0.5, dtype='float32')
            shape = paddle.shape(data2)
            with paddle.static.device_guard('cpu'):
                shape = paddle.slice(shape, axes=[0], starts=[0], ends=[4])
                with paddle.static.device_guard('gpu'):
                    out = paddle.crop(data1, shape=shape)
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'slice':
                self.assertEqual(op.desc.attr(device_attr_name), 'cpu')
            if op.type == 'crop_tensor':
                self.assertEqual(op.desc.attr(device_attr_name), 'gpu')
        execute(main_program, startup_program)

    def test_device_guard_with_id(self):
        if False:
            while True:
                i = 10
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data1 = paddle.full(shape=[1, 3, 8, 8], fill_value=0.5, dtype='float32')
            data2 = paddle.full(shape=[1, 3, 5, 5], fill_value=0.5, dtype='float32')
            shape = paddle.shape(data2)
            with paddle.static.device_guard('cpu'):
                shape = paddle.slice(shape, axes=[0], starts=[0], ends=[4])
                with paddle.static.device_guard('gpu:1'):
                    out = paddle.crop(data1, shape=shape)
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'slice':
                self.assertEqual(op.desc.attr(device_attr_name), 'cpu')
            if op.type == 'crop_tensor':
                self.assertEqual(op.desc.attr(device_attr_name), 'gpu:1')
        execute(main_program, startup_program)

    def test_cpu_only_op(self):
        if False:
            i = 10
            return i + 15
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.full(shape=[2, 255, 13, 13], fill_value=0.3, dtype='float32')
            gt_box = paddle.full(shape=[2, 6, 4], fill_value=0.5, dtype='float32')
            gt_label = paddle.full(shape=[2, 6], fill_value=1.0, dtype='int32')
            gt_score = paddle.full(shape=[2, 6], fill_value=0.5, dtype='float32')
            anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
            anchor_mask = [0, 1, 2]
            with paddle.static.device_guard('gpu'):
                loss = paddle.vision.ops.yolo_loss(x=x, gt_box=gt_box, gt_label=gt_label, gt_score=gt_score, anchors=anchors, anchor_mask=anchor_mask, class_num=80, ignore_thresh=0.7, downsample_ratio=32)
        execute(main_program, startup_program)

    def test_without_kernel_op(self):
        if False:
            return 10
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.full(shape=[1], dtype='int64', fill_value=0)
            loop_len = paddle.full(shape=[1], dtype='int64', fill_value=10)
            cond = paddle.less_than(x=i, y=loop_len)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                with paddle.static.device_guard('cpu'):
                    while_op = paddle.static.nn.control_flow.While(cond=cond)
                    with while_op.block():
                        i = paddle.increment(x=i, value=1)
                        paddle.assign(paddle.less_than(x=i, y=loop_len), cond)
        warning = 'The Op(while) is not support to set device.'
        warning_num = get_vaild_warning_num(warning, w)
        assert warning_num == 1
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'while':
                self.assertEqual(op.desc.attr(device_attr_name), '')
        execute(main_program, startup_program)

    def test_error(self):
        if False:
            print('Hello World!')

        def device_attr():
            if False:
                while True:
                    i = 10
            with paddle.static.device_guard('cpu1'):
                out = paddle.full(shape=[1], fill_value=0.2, dtype='float32')

        def device_attr2():
            if False:
                while True:
                    i = 10
            with paddle.static.device_guard('cpu:1'):
                out = paddle.full(shape=[1], fill_value=0.2, dtype='float32')
        self.assertRaises(ValueError, device_attr)
        self.assertRaises(ValueError, device_attr2)

    def test_op_descs_device_attr(self):
        if False:
            for i in range(10):
                print('nop')
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data1 = paddle.static.data(name='data_1', shape=[4, 2], dtype='float32')
            label = paddle.static.data(name='label', shape=[4, 1], dtype='int64')
            fc1 = paddle.static.nn.fc(x=data1, size=10)
            fc2 = paddle.static.nn.fc(x=fc1, size=10)
            with paddle.static.device_guard('gpu'):
                out = paddle.nn.functional.softmax_with_cross_entropy(logits=fc1 + fc2, label=label)
                loss = paddle.mean(out)
                opt = paddle.optimizer.SGD(0.1)
                opt.minimize(loss)
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            self.assertEqual(True, op.desc.has_attr(device_attr_name))
            if op.desc == 'fill_constant':
                self.assertEqual(op.desc.attr(device_attr_name), 'gpu')
if __name__ == '__main__':
    unittest.main()
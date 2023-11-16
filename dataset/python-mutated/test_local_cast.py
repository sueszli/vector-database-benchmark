import unittest
import paddle
from paddle.jit.dy2static.program_translator import ProgramTranslator
from paddle.static.amp.fp16_utils import DEFAULT_AMP_OPTIONS, AmpOptions, prepare_op_amp_options
GLOBAL_ENABLE_AMP_OPTIONS = DEFAULT_AMP_OPTIONS
GLOBAL_DISABLE_AMP_OPTIONS = AmpOptions(enable=False, custom_black_list=DEFAULT_AMP_OPTIONS.custom_black_list, custom_white_list=DEFAULT_AMP_OPTIONS.custom_white_list, level=DEFAULT_AMP_OPTIONS.level, dtype=DEFAULT_AMP_OPTIONS.dtype, use_promote=DEFAULT_AMP_OPTIONS.use_promote)

class LocalAutoCastLayer1(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._fc = paddle.nn.Linear(10, 10)

    @paddle.jit.to_static(full_graph=True)
    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self._fc(x)
        y = self._fc(x) * 2
        with paddle.amp.auto_cast(False):
            x = x.astype('float32')
            y = y.astype('float32')
            if x[0][0] > 1:
                x = x + y
            else:
                x = x - y
                x = x * 2
        return x + 1

class LocalAutoCastLayer2(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._fc = paddle.nn.Linear(10, 10)

    @paddle.jit.to_static(full_graph=True)
    def forward(self, x):
        if False:
            i = 10
            return i + 15
        with paddle.amp.auto_cast(False):
            x = x.astype('float32')
            x = self._fc(x)
            y = self._fc(x) * 2
        if x[0][0] > 1:
            x = x + y
        else:
            x = x - y
            x = x * 2
        return x + 1

class LocalAutoCastLayer3(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self._fc = paddle.nn.Linear(10, 10)

    @paddle.jit.to_static(full_graph=True)
    def forward(self, x):
        if False:
            return 10
        with paddle.amp.auto_cast(True):
            x = x.astype('float32')
            x = self._fc(x)
            y = self._fc(x) * 2
        if x[0][0] > 1:
            x = x + y
        else:
            x = x - y
            x = x * 2
        return x + 1

class TestLocalCast(unittest.TestCase):

    def get_auto_cast_ops_info_from_program(self, program):
        if False:
            for i in range(10):
                print('nop')
        auto_cast_ops_info = []
        for block in program.blocks:
            current_block_should_auto_cast = []
            auto_cast_ops_info.append(current_block_should_auto_cast)
            for op in block.ops:
                current_block_should_auto_cast.append(op.amp_options.enable)
        return auto_cast_ops_info

    def should_auto_cast_for_each_ops(self, layer, input, global_amp_options):
        if False:
            print('Hello World!')
        (concrete_program, _) = layer.forward.get_concrete_program(input)
        program = concrete_program.main_program
        prepare_op_amp_options(program, ProgramTranslator.get_instance()._amp_records, global_amp_options)
        auto_cast_ops_info = self.get_auto_cast_ops_info_from_program(program)
        paddle.enable_static()
        cloned_program = program.clone()
        paddle.disable_static()
        cloned_auto_cast_ops_info = self.get_auto_cast_ops_info_from_program(cloned_program)
        self.assertEqual(auto_cast_ops_info, cloned_auto_cast_ops_info)
        return auto_cast_ops_info

    def test_should_auto_cast_1(self):
        if False:
            for i in range(10):
                print('nop')
        layer = LocalAutoCastLayer1()
        input = paddle.randn([10, 10])
        expected = [[True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True], [False, False], [False, False, False]]
        actual = self.should_auto_cast_for_each_ops(layer, input, GLOBAL_ENABLE_AMP_OPTIONS)
        self.assertEqual(expected, actual)

    def test_should_auto_cast_2(self):
        if False:
            for i in range(10):
                print('nop')
        layer = LocalAutoCastLayer2()
        input = paddle.randn([10, 10])
        expected = [[False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True], [True, True], [True, True, True]]
        actual = self.should_auto_cast_for_each_ops(layer, input, GLOBAL_ENABLE_AMP_OPTIONS)
        self.assertEqual(expected, actual)

    def test_should_auto_cast_3(self):
        if False:
            i = 10
            return i + 15
        layer = LocalAutoCastLayer3()
        input = paddle.randn([10, 10])
        expected = [[True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False], [False, False], [False, False, False]]
        actual = self.should_auto_cast_for_each_ops(layer, input, GLOBAL_DISABLE_AMP_OPTIONS)
        self.assertEqual(expected, actual)
if __name__ == '__main__':
    unittest.main()
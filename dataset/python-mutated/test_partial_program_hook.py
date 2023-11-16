import unittest
from dygraph_to_static_utils_new import Dy2StTestBase
import paddle
from paddle.base import core
from paddle.jit.api import ENV_ENABLE_SOT
from paddle.jit.dy2static import partial_program, program_translator

class TestPartiaProgramLayerHook(Dy2StTestBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        ENV_ENABLE_SOT.set(False)
        self._hook = partial_program.PartialProgramLayerHook()

    def test_before_append_backward(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNone(self._hook.before_append_backward(None))

    def test_after_append_backward(self):
        if False:
            return 10
        self.assertIsNone(self._hook.after_append_backward(None, 0))

    def test_after_infer(self):
        if False:
            print('Hello World!')
        self.assertIsNone(self._hook.after_infer(None))

class TestPrimHook(Dy2StTestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ENV_ENABLE_SOT.set(False)
        core._set_prim_all_enabled(False)

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return paddle.nn.functional.dropout(paddle.rand((1,)))
        (concrete_program, partial_program) = paddle.jit.to_static(f).get_concrete_program()
        self._hook = program_translator.PrimHooker(concrete_program.main_program, None)
        self._forward = partial_program.forward_program
        self._whole = partial_program._train_program
        core._set_prim_all_enabled(True)

    def tearDown(self):
        if False:
            while True:
                i = 10
        core._set_prim_all_enabled(False)

    def test_before_append_backward(self):
        if False:
            return 10
        self._hook.before_append_backward(self._forward)
        self.assertNotIn('dropout', tuple((op.type for op in self._forward.blocks[0].ops)))

    def test_after_append_backward(self):
        if False:
            i = 10
            return i + 15
        self._hook.after_append_backward(self._whole, 0)
        self.assertNotIn('dropout_grad', tuple((op.type for op in self._whole.blocks[0].ops)))
if __name__ == '__main__':
    unittest.main()
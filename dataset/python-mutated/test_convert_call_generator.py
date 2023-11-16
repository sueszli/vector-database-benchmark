import unittest
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_legacy_and_pir
import paddle
from paddle.jit import to_static
from paddle.jit.dy2static.convert_call_func import translator_logger

def dyfunc_generator():
    if False:
        print('Hello World!')
    for i in range(100):
        yield paddle.to_tensor([i] * 10)

def main_func():
    if False:
        while True:
            i = 10
    'Error will raise, but we only report a warning not intercept'
    for i in dyfunc_generator():
        print(i)

class TestConvertGenerator(Dy2StTestBase):

    @test_ast_only
    @test_legacy_and_pir
    def test_raise_error(self):
        if False:
            while True:
                i = 10
        translator_logger.verbosity_level = 1
        with self.assertLogs(translator_logger.logger_name, level='WARNING') as cm:
            to_static(main_func)()
            self.assertRegex(cm.output[0], "Your function:`dyfunc_generator` doesn't support to transform to static function because it is a generator function")
if __name__ == '__main__':
    unittest.main()
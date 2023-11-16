import unittest
import jittor as jt
import numpy as np

class TestErrorMsg(unittest.TestCase):

    def test_error_msg(self):
        if False:
            return 10
        a = jt.array([3, 2, 1])
        b = jt.code(a.shape, a.dtype, [a], cpu_header='\n                #include <algorithm>\n                @alias(a, in0)\n                @alias(b, out)\n            ', cpu_src='\n                for (int i=0; i<a_shape0; i++)\n                    @b(i) = @a(i);\n                std::sort(&@b(0), &@b(in0_shape0));\n                throw std::runtime_error("???");\n            ')
        msg = ''
        try:
            print(b)
        except Exception as e:
            msg = str(e)
        assert '[Reason]: ???' in msg
        assert '[Input]: int32[3,]' in msg
        assert '[OP TYPE]: code' in msg
        assert '[Async Backtrace]:' in msg

    @jt.flag_scope(trace_py_var=3)
    def test_error_msg_trace_py_var(self):
        if False:
            for i in range(10):
                print('nop')
        a = jt.array([3, 2, 1])
        b = jt.code(a.shape, a.dtype, [a], cpu_header='\n                #include <algorithm>\n                @alias(a, in0)\n                @alias(b, out)\n            ', cpu_src='\n                for (int i=0; i<a_shape0; i++)\n                    @b(i) = @a(i);\n                std::sort(&@b(0), &@b(in0_shape0));\n                throw std::runtime_error("???");\n            ')
        msg = ''
        try:
            print(b)
        except Exception as e:
            msg = str(e)
        print(msg)
        assert '[Reason]: ???' in msg
        assert '[Input]: int32[3,]' in msg
        assert '[OP TYPE]: code' in msg
        assert '[Async Backtrace]:' in msg
        assert 'test_error_msg.py:' in msg
if __name__ == '__main__':
    unittest.main()
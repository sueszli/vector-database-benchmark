"""Tests for `tensorflow::FunctionParameterCanonicalizer`."""
from tensorflow.python.platform import test
from tensorflow.python.util import _function_parameter_canonicalizer_binding_for_test

class FunctionParameterCanonicalizerTest(test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(FunctionParameterCanonicalizerTest, self).setUp()
        self._matmul_func = _function_parameter_canonicalizer_binding_for_test.FunctionParameterCanonicalizer(['a', 'b', 'transpose_a', 'transpose_b', 'adjoint_a', 'adjoint_b', 'a_is_sparse', 'b_is_sparse', 'name'], (False, False, False, False, False, False, None))

    def testPosOnly(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self._matmul_func.canonicalize(2, 3), [2, 3, False, False, False, False, False, False, None])

    def testPosOnly2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self._matmul_func.canonicalize(2, 3, True, False, True), [2, 3, True, False, True, False, False, False, None])

    def testPosAndKwd(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self._matmul_func.canonicalize(2, 3, transpose_a=True, name='my_matmul'), [2, 3, True, False, False, False, False, False, 'my_matmul'])

    def testPosAndKwd2(self):
        if False:
            return 10
        self.assertEqual(self._matmul_func.canonicalize(2, b=3), [2, 3, False, False, False, False, False, False, None])

    def testMissingPos(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'Missing required positional argument'):
            self._matmul_func.canonicalize(2)

    def testMissingPos2(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, 'Missing required positional argument'):
            self._matmul_func.canonicalize(transpose_a=True, transpose_b=True, adjoint_a=True)

    def testTooManyArgs(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'Too many arguments were given. Expected 9 but got 10.'):
            self._matmul_func.canonicalize(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    def testInvalidKwd(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, 'Got an unexpected keyword argument'):
            self._matmul_func.canonicalize(2, 3, hohoho=True)

    def testDuplicatedArg(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, "Got multiple values for argument 'b'"):
            self._matmul_func.canonicalize(2, 3, False, b=4)

    def testDuplicatedArg2(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, "Got multiple values for argument 'transpose_a'"):
            self._matmul_func.canonicalize(2, 3, False, transpose_a=True)

    def testKwargNotInterned(self):
        if False:
            print('Hello World!')
        func = _function_parameter_canonicalizer_binding_for_test.FunctionParameterCanonicalizer(['long_parameter_name'], ())
        kwargs = dict([('_'.join(['long', 'parameter', 'name']), 5)])
        func.canonicalize(**kwargs)
if __name__ == '__main__':
    test.main()
"""Tests for utils.py."""
from absl.testing import parameterized
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.platform import test

class UtilsTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(UtilsTest, self).setUp()
        self._old_np_doc_form = np_utils.get_np_doc_form()
        self._old_is_sig_mismatch_an_error = np_utils.is_sig_mismatch_an_error()

    def tearDown(self):
        if False:
            print('Hello World!')
        np_utils.set_np_doc_form(self._old_np_doc_form)
        np_utils.set_is_sig_mismatch_an_error(self._old_is_sig_mismatch_an_error)
        super(UtilsTest, self).tearDown()

    def testNpDocInlined(self):
        if False:
            for i in range(10):
                print('nop')

        def np_fun(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            'np_fun docstring.'
            return
        np_utils.set_np_doc_form('inlined')

        @np_utils.np_doc(None, np_fun=np_fun, unsupported_params=['x'])
        def f(x, z):
            if False:
                return 10
            'f docstring.'
            return
        expected = "TensorFlow variant of NumPy's `np_fun`.\n\nUnsupported arguments: `x`, `y`.\n\nf docstring.\n\nDocumentation for `numpy.np_fun`:\n\nnp_fun docstring."
        self.assertEqual(expected, f.__doc__)

    @parameterized.named_parameters([(version, version, link) for (version, link) in [('dev', 'https://numpy.org/devdocs/reference/generated/numpy.np_fun.html'), ('stable', 'https://numpy.org/doc/stable/reference/generated/numpy.np_fun.html'), ('1.16', 'https://numpy.org/doc/1.16/reference/generated/numpy.np_fun.html')]])
    def testNpDocLink(self, version, link):
        if False:
            for i in range(10):
                print('nop')

        def np_fun(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            'np_fun docstring.'
            return
        np_utils.set_np_doc_form(version)

        @np_utils.np_doc(None, np_fun=np_fun, unsupported_params=['x'])
        def f(x, z):
            if False:
                print('Hello World!')
            'f docstring.'
            return
        expected = "TensorFlow variant of NumPy's `np_fun`.\n\nUnsupported arguments: `x`, `y`.\n\nf docstring.\n\nSee the NumPy documentation for [`numpy.np_fun`](%s)."
        expected = expected % link
        self.assertEqual(expected, f.__doc__)

    @parameterized.parameters([None, 1, 'a', '1a', '1.1a', '1.1.1a'])
    def testNpDocInvalid(self, invalid_flag):
        if False:
            while True:
                i = 10

        def np_fun(x, y, z):
            if False:
                while True:
                    i = 10
            'np_fun docstring.'
            return
        np_utils.set_np_doc_form(invalid_flag)

        @np_utils.np_doc(None, np_fun=np_fun, unsupported_params=['x'])
        def f(x, z):
            if False:
                for i in range(10):
                    print('nop')
            'f docstring.'
            return
        expected = "TensorFlow variant of NumPy's `np_fun`.\n\nUnsupported arguments: `x`, `y`.\n\nf docstring.\n\n"
        self.assertEqual(expected, f.__doc__)

    def testNpDocName(self):
        if False:
            print('Hello World!')
        np_utils.set_np_doc_form('inlined')

        @np_utils.np_doc('foo')
        def f():
            if False:
                for i in range(10):
                    print('nop')
            'f docstring.'
            return
        expected = "TensorFlow variant of NumPy's `foo`.\n\nf docstring.\n\n"
        self.assertEqual(expected, f.__doc__)

    def testDtypeOfTensorLikeClass(self):
        if False:
            for i in range(10):
                print('nop')

        class TensorLike:

            def __init__(self, dtype):
                if False:
                    while True:
                        i = 10
                self._dtype = dtype

            @property
            def is_tensor_like(self):
                if False:
                    i = 10
                    return i + 15
                return True

            @property
            def dtype(self):
                if False:
                    i = 10
                    return i + 15
                return self._dtype
        t = TensorLike(dtypes.float32)
        self.assertEqual(np_utils._maybe_get_dtype(t), dtypes.float32)

    def testSigMismatchIsError(self):
        if False:
            i = 10
            return i + 15
        'Tests that signature mismatch is an error (when configured so).'
        if not np_utils._supports_signature():
            self.skipTest('inspect.signature not supported')
        np_utils.set_is_sig_mismatch_an_error(True)

        def np_fun(x, y=1, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return
        with self.assertRaisesRegex(TypeError, 'Cannot find parameter'):

            @np_utils.np_doc(None, np_fun=np_fun)
            def f1(a):
                if False:
                    print('Hello World!')
                return
        with self.assertRaisesRegex(TypeError, 'is of kind'):

            @np_utils.np_doc(None, np_fun=np_fun)
            def f2(x, kwargs):
                if False:
                    print('Hello World!')
                return
        with self.assertRaisesRegex(TypeError, 'Parameter y should have a default value'):

            @np_utils.np_doc(None, np_fun=np_fun)
            def f3(x, y):
                if False:
                    print('Hello World!')
                return

    def testSigMismatchIsNotError(self):
        if False:
            return 10
        'Tests that signature mismatch is not an error (when configured so).'
        np_utils.set_is_sig_mismatch_an_error(False)

        def np_fun(x, y=1, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return

        @np_utils.np_doc(None, np_fun=np_fun)
        def f1(a):
            if False:
                for i in range(10):
                    print('nop')
            return

        def f2(x, kwargs):
            if False:
                return 10
            return

        @np_utils.np_doc(None, np_fun=np_fun)
        def f3(x, y):
            if False:
                while True:
                    i = 10
            return
if __name__ == '__main__':
    test.main()
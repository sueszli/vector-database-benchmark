import importlib
import os
import unittest
import xdoctest
from sampcd_processor import Xdoctester
from sampcd_processor_utils import TestResult as _TestResult, get_api_md5, get_incrementapi, get_test_results

def _clear_environ():
    if False:
        return 10
    for k in {'CPU', 'GPU', 'XPU', 'DISTRIBUTED'}:
        if k in os.environ:
            del os.environ[k]

class Test_TestResult(unittest.TestCase):

    def test_good_result(self):
        if False:
            print('Hello World!')
        r = _TestResult(name='good', passed=True)
        self.assertTrue(r.passed)
        self.assertFalse(r.failed)
        r = _TestResult(name='good', passed=True, failed=False)
        self.assertTrue(r.passed)
        self.assertFalse(r.failed)
        r = _TestResult(name='good', passed=False, failed=True)
        self.assertFalse(r.passed)
        self.assertTrue(r.failed)
        r = _TestResult(name='good', passed=True, nocode=False, time=10)
        self.assertTrue(r.passed)
        self.assertFalse(r.nocode)
        r = _TestResult(name='good', passed=True, timeout=False, time=10, test_msg='ok', extra_info=None)
        self.assertTrue(r.passed)
        self.assertFalse(r.timeout)

    def test_bad_result(self):
        if False:
            print('Hello World!')
        r = _TestResult(name='bad', passed=True, failed=True)
        self.assertTrue(r.passed)
        self.assertTrue(r.failed)
        r = _TestResult(name='bad')
        self.assertFalse(r.passed)
        self.assertTrue(r.failed)
        with self.assertRaises(KeyError):
            r = _TestResult(name='good', passed=True, bad=True)

class Test_get_api_md5(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.api_pr_spec_filename = os.path.abspath(os.path.join(os.getcwd(), '..', 'paddle/fluid/API_PR.spec'))
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write('\n'.join(["paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))", "paddle.two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55two'))", "paddle.three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6cthree'))", "paddle.four_plus_four (paddle.four_plus_four, ('document', 'ff0f188c95030158cc6398d2a6c5four'))", "paddle.five_plus_five (ArgSpec(), ('document', 'ff0f188c95030158cc6398d2a6c5five'))"]))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        os.remove(self.api_pr_spec_filename)

    def test_get_api_md5(self):
        if False:
            while True:
                i = 10
        res = get_api_md5('paddle/fluid/API_PR.spec')
        self.assertEqual('ff0f188c95030158cc6398d2a6c55one', res['paddle.one_plus_one'])
        self.assertEqual('ff0f188c95030158cc6398d2a6c55two', res['paddle.two_plus_two'])
        self.assertEqual('ff0f188c95030158cc6398d2a6cthree', res['paddle.three_plus_three'])
        self.assertEqual('ff0f188c95030158cc6398d2a6c5four', res['paddle.four_plus_four'])
        self.assertEqual('ff0f188c95030158cc6398d2a6c5five', res['paddle.five_plus_five'])

class Test_get_incrementapi(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.api_pr_spec_filename = os.path.abspath(os.path.join(os.getcwd(), '..', 'paddle/fluid/API_PR.spec'))
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write('\n'.join(["paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))", "paddle.two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55two'))", "paddle.three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6cthree'))", "paddle.four_plus_four (paddle.four_plus_four, ('document', 'ff0f188c95030158cc6398d2a6c5four'))"]))
        self.api_dev_spec_filename = os.path.abspath(os.path.join(os.getcwd(), '..', 'paddle/fluid/API_DEV.spec'))
        with open(self.api_dev_spec_filename, 'w') as f:
            f.write('\n'.join(["paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))"]))
        self.api_diff_spec_filename = os.path.abspath(os.path.join(os.getcwd(), 'dev_pr_diff_api.spec'))

    def tearDown(self):
        if False:
            return 10
        os.remove(self.api_pr_spec_filename)
        os.remove(self.api_dev_spec_filename)
        os.remove(self.api_diff_spec_filename)

    def test_it(self):
        if False:
            print('Hello World!')
        get_incrementapi()
        with open(self.api_diff_spec_filename, 'r') as f:
            lines = f.readlines()
            self.assertCountEqual(['paddle.two_plus_two\n', 'paddle.three_plus_three\n', 'paddle.four_plus_four\n'], lines)

class TestXdoctester(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.maxDiff = None

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        doctester = Xdoctester()
        self.assertEqual(doctester.debug, False)
        self.assertEqual(doctester.style, 'freeform')
        self.assertEqual(doctester.target, 'codeblock')
        self.assertEqual(doctester.mode, 'native')
        doctester = Xdoctester(analysis='static')
        self.assertEqual(doctester.config['analysis'], 'static')

    def test_convert_directive(self):
        if False:
            i = 10
            return i + 15
        doctester = Xdoctester()
        docstring_input = '# doctest: -SKIP\n'
        docstring_output = doctester.convert_directive(docstring_input)
        docstring_target = '# xdoctest: -SKIP\n'
        self.assertEqual(docstring_output, docstring_target)
        docstring_input = '# doctest: +SKIP("skip this test...")\n'
        docstring_output = doctester.convert_directive(docstring_input)
        docstring_target = '# xdoctest: +SKIP("skip this test...")\n'
        self.assertEqual(docstring_output, docstring_target)
        docstring_input = "\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> # doctest: +SKIP('skip')\n                >>> print(1+1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: -REQUIRES(env:GPU)\n                    >>> print(1-1)\n                    0\n\n                .. code-block:: python\n                    :name: code-example-2\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:GPU, env:XPU, env: DISTRIBUTED)\n                    >>> print(1-1)\n                    0\n            "
        docstring_output = doctester.convert_directive(docstring_input)
        docstring_target = "\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> # xdoctest: +SKIP('skip')\n                >>> print(1+1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # xdoctest: -REQUIRES(env:GPU)\n                    >>> print(1-1)\n                    0\n\n                .. code-block:: python\n                    :name: code-example-2\n\n                    this is some blabla...\n\n                    >>> # xdoctest: +REQUIRES(env:GPU, env:XPU, env: DISTRIBUTED)\n                    >>> print(1-1)\n                    0\n            "
        self.assertEqual(docstring_output, docstring_target)

    def test_prepare(self):
        if False:
            print('Hello World!')
        doctester = Xdoctester()
        _clear_environ()
        test_capacity = {'cpu'}
        doctester.prepare(test_capacity)
        self.assertTrue(os.environ['CPU'])
        self.assertFalse(os.environ.get('GPU'))
        _clear_environ()
        test_capacity = {'cpu', 'gpu'}
        doctester.prepare(test_capacity)
        self.assertTrue(os.environ['CPU'])
        self.assertTrue(os.environ['GPU'])
        self.assertFalse(os.environ.get('cpu'))
        self.assertFalse(os.environ.get('gpu'))
        self.assertFalse(os.environ.get('XPU'))
        _clear_environ()

class TestGetTestResults(unittest.TestCase):

    def test_global_exec(self):
        if False:
            return 10
        _clear_environ()
        docstrings_to_test = {'before_set_default': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> a = paddle.to_tensor(.2)\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,\n                    [0.20000000])\n            ', 'set_default': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.set_default_dtype('float64')\n                    >>> a = paddle.to_tensor(.2)\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float64, place=Place(cpu), stop_gradient=True,\n                    [0.20000000])\n            ", 'after_set_default': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> a = paddle.to_tensor(.2)\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,\n                    [0.20000000])\n            '}
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock', global_exec='\\n'.join(['import paddle', "paddle.device.set_device('cpu')"]))
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 3)
        (tr_0, tr_1, tr_2) = test_results
        self.assertIn('before_set_default', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertIn('set_default', tr_1.name)
        self.assertTrue(tr_1.passed)
        self.assertIn('after_set_default', tr_2.name)
        self.assertTrue(tr_2.passed)
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 3)
        (tr_0, tr_1, tr_2) = test_results
        self.assertIn('before_set_default', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertIn('set_default', tr_1.name)
        self.assertTrue(tr_1.passed)
        self.assertIn('after_set_default', tr_2.name)
        self.assertTrue(tr_2.passed)
        docstrings_to_test = {'before_enable_static': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> print(paddle.in_dynamic_mode())\n                    True\n            ', 'enable_static': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.enable_static()\n                    >>> print(paddle.in_dynamic_mode())\n                    False\n            ', 'after_enable_static': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> print(paddle.in_dynamic_mode())\n                    True\n            '}
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock', global_exec='\\n'.join(['import paddle', "paddle.device.set_device('cpu')"]))
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 3)
        (tr_0, tr_1, tr_2) = test_results
        self.assertIn('before_enable_static', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertIn('enable_static', tr_1.name)
        self.assertTrue(tr_1.passed)
        self.assertIn('after_enable_static', tr_2.name)
        self.assertTrue(tr_2.passed)
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 3)
        (tr_0, tr_1, tr_2) = test_results
        self.assertIn('before_enable_static', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertIn('enable_static', tr_1.name)
        self.assertTrue(tr_1.passed)
        self.assertIn('after_enable_static', tr_2.name)
        self.assertTrue(tr_2.passed)

    def test_patch_xdoctest(self):
        if False:
            for i in range(10):
                print('nop')
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)
        docstrings_to_test = {'gpu_to_gpu': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('gpu')\n                    >>> a = paddle.to_tensor(.2)\n                    >>> # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [0.20000000])\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,\n                    [0.20000000])\n\n            ", 'cpu_to_cpu': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('cpu')\n                    >>> a = paddle.to_tensor(.2)\n                    >>> # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [0.20000000])\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,\n                    [0.20000000])\n\n            ", 'gpu_to_cpu': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('gpu')\n                    >>> a = paddle.to_tensor(.2)\n                    >>> # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [0.20000000])\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,\n                    [0.20000000])\n\n            ", 'cpu_to_gpu': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('cpu')\n                    >>> a = paddle.to_tensor(.2)\n                    >>> # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [0.20000000])\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,\n                    [0.20000000])\n            ", 'gpu_to_cpu_array': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('gpu')\n                    >>> a = paddle.to_tensor([[1,2,3], [2,3,4], [3,4,5]])\n                    >>> # Tensor(shape=[3, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,\n                    >>> # [[1, 2, 3],\n                    >>> # [2, 3, 4],\n                    >>> # [3, 4, 5]])\n                    >>> print(a)\n                    Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,\n                    [[1, 2, 3],\n                    [2, 3, 4],\n                    [3, 4, 5]])\n            ", 'cpu_to_gpu_array': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('cpu')\n                    >>> a = paddle.to_tensor([[1,2,3], [2,3,4], [3,4,5]])\n                    >>> # Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,\n                    >>> # [[1, 2, 3],\n                    >>> # [2, 3, 4],\n                    >>> # [3, 4, 5]])\n                    >>> print(a)\n                    Tensor(shape=[3, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,\n                    [[1, 2, 3],\n                    [2, 3, 4],\n                    [3, 4, 5]])\n            "}
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 6)
        (tr_0, tr_1, tr_2, tr_3, tr_4, tr_5) = test_results
        self.assertIn('gpu_to_gpu', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertIn('cpu_to_cpu', tr_1.name)
        self.assertTrue(tr_1.passed)
        self.assertIn('gpu_to_cpu', tr_2.name)
        self.assertTrue(tr_2.passed)
        self.assertIn('cpu_to_gpu', tr_3.name)
        self.assertTrue(tr_3.passed)
        self.assertIn('gpu_to_cpu_array', tr_4.name)
        self.assertTrue(tr_4.passed)
        self.assertIn('cpu_to_gpu_array', tr_5.name)
        self.assertTrue(tr_5.passed)
        importlib.reload(xdoctest.checker)
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock', patch_tensor_place=False)
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 6)
        (tr_0, tr_1, tr_2, tr_3, tr_4, tr_5) = test_results
        self.assertIn('gpu_to_gpu', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertIn('cpu_to_cpu', tr_1.name)
        self.assertTrue(tr_1.passed)
        self.assertIn('gpu_to_cpu', tr_2.name)
        self.assertFalse(tr_2.passed)
        self.assertIn('cpu_to_gpu', tr_3.name)
        self.assertFalse(tr_3.passed)
        self.assertIn('gpu_to_cpu_array', tr_4.name)
        self.assertFalse(tr_4.passed)
        self.assertIn('cpu_to_gpu_array', tr_5.name)
        self.assertFalse(tr_5.passed)
        importlib.reload(xdoctest.checker)
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)
        docstrings_to_test = {'gpu_to_gpu': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('gpu')\n                    >>> a = paddle.to_tensor(.123456789)\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,\n                    [0.123456780])\n\n            ", 'cpu_to_cpu': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('cpu')\n                    >>> a = paddle.to_tensor(.123456789)\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,\n                    [0.123456780])\n\n            ", 'gpu_to_cpu': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('gpu')\n                    >>> a = paddle.to_tensor(.123456789)\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,\n                    [0.123456780])\n\n            ", 'cpu_to_gpu': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('cpu')\n                    >>> a = paddle.to_tensor(.123456789)\n                    >>> print(a)\n                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,\n                    [0.123456780])\n            ", 'gpu_to_cpu_array': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('gpu')\n                    >>> a = paddle.to_tensor([[1.123456789 ,2,3], [2,3,4], [3,4,5]])\n                    >>> print(a)\n                    Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,\n                    [[1.123456780, 2., 3.],\n                    [2., 3., 4.],\n                    [3., 4., 5.]])\n            ", 'cpu_to_gpu_array': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('cpu')\n                    >>> a = paddle.to_tensor([[1.123456789,2,3], [2,3,4], [3,4,5]])\n                    >>> print(a)\n                    Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,\n                    [[1.123456780, 2., 3.],\n                    [2., 3., 4.],\n                    [3., 4., 5.]])\n            ", 'mass_array': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('gpu')\n                    >>> a = paddle.to_tensor(\n                    ... [[1.123456780, 2., -3, .3],\n                    ... [2, 3, +4., 1.2+10.34e-5j],\n                    ... [3, 5.e-3, 1e2, 3e-8]]\n                    ... )\n                    >>> # Tensor(shape=[3, 4], dtype=complex64, place=Place(gpu:0), stop_gradient=True,\n                    >>> #       [[ (1.1234568357467651+0j)                    ,\n                    >>> #          (2+0j)                                     ,\n                    >>> #         (-3+0j)                                     ,\n                    >>> #          (0.30000001192092896+0j)                   ],\n                    >>> #        [ (2+0j)                                     ,\n                    >>> #          (3+0j)                                     ,\n                    >>> #          (4+0j)                                     ,\n                    >>> #         (1.2000000476837158+0.00010340000153519213j)],\n                    >>> #        [ (3+0j)                                     ,\n                    >>> #          (0.004999999888241291+0j)                  ,\n                    >>> #          (100+0j)                                   ,\n                    >>> #          (2.999999892949745e-08+0j)                 ]])\n                    >>> print(a)\n                    Tensor(shape=[3, 4], dtype=complex64, place=Place(AAA), stop_gradient=True,\n                        [[ (1.123456+0j),\n                            (2+0j),\n                            (-3+0j),\n                            (0.3+0j)],\n                            [ (2+0j),\n                            (3+0j),\n                            (4+0j),\n                            (1.2+0.00010340j)],\n                            [ (3+0j),\n                            (0.00499999+0j),\n                            (100+0j),\n                            (2.999999e-08+0j)]])\n            ", 'float_array': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('cpu')\n                    >>> x = [[2, 3, 4], [7, 8, 9]]\n                    >>> x = paddle.to_tensor(x, dtype='float32')\n                    >>> print(paddle.log(x))\n                    Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,\n                    [[0.69314718, 1.09861231, 1.38629436],\n                        [1.94591010, 2.07944155, 2.19722462]])\n\n            ", 'float_array_diff': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import paddle\n                    >>> paddle.device.set_device('cpu')\n                    >>> x = [[2, 3, 4], [7, 8, 9]]\n                    >>> x = paddle.to_tensor(x, dtype='float32')\n                    >>> print(paddle.log(x))\n                    Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,\n                        [[0.69314712, 1.09861221, 1.386294],\n                        [1.94591032, 2.07944156, 2.1972246]])\n\n            ", 'float_begin': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> print(7.0)\n                    7.\n\n            ', 'float_begin_long': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> print(7.0000023)\n                    7.0000024\n\n            ', 'float_begin_more': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> print(7.0, 5., 6.123456)\n                    7.0 5.0 6.123457\n\n            ', 'float_begin_more_diff': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> print(7.0, 5., 6.123456)\n                    7.0 5.0 6.123457\n\n            ', 'float_begin_more_brief': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> print(7.0, 5., 6.123456)\n                    7. 5. 6.123457\n\n            ', 'float_begin_fail': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> print(7.0100023)\n                    7.0000024\n\n            '}
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 15)
        (tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8, tr_9, tr_10, tr_11, tr_12, tr_13, tr_14) = test_results
        self.assertIn('gpu_to_gpu', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertIn('cpu_to_cpu', tr_1.name)
        self.assertTrue(tr_1.passed)
        self.assertIn('gpu_to_cpu', tr_2.name)
        self.assertTrue(tr_2.passed)
        self.assertIn('cpu_to_gpu', tr_3.name)
        self.assertTrue(tr_3.passed)
        self.assertIn('gpu_to_cpu_array', tr_4.name)
        self.assertTrue(tr_4.passed)
        self.assertIn('cpu_to_gpu_array', tr_5.name)
        self.assertTrue(tr_5.passed)
        self.assertIn('mass_array', tr_6.name)
        self.assertTrue(tr_6.passed)
        self.assertIn('float_array', tr_7.name)
        self.assertTrue(tr_7.passed)
        self.assertIn('float_array_diff', tr_8.name)
        self.assertTrue(tr_8.passed)
        self.assertIn('float_begin', tr_9.name)
        self.assertTrue(tr_9.passed)
        self.assertIn('float_begin_long', tr_10.name)
        self.assertTrue(tr_10.passed)
        self.assertIn('float_begin_more', tr_11.name)
        self.assertTrue(tr_11.passed)
        self.assertIn('float_begin_more_diff', tr_12.name)
        self.assertTrue(tr_12.passed)
        self.assertIn('float_begin_more_brief', tr_13.name)
        self.assertTrue(tr_13.passed)
        self.assertIn('float_begin_fail', tr_14.name)
        self.assertFalse(tr_14.passed)
        importlib.reload(xdoctest.checker)
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock', patch_float_precision=None)
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 15)
        (tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8, tr_9, tr_10, tr_11, tr_12, tr_13, tr_14) = test_results
        self.assertIn('gpu_to_gpu', tr_0.name)
        self.assertFalse(tr_0.passed)
        self.assertIn('cpu_to_cpu', tr_1.name)
        self.assertFalse(tr_1.passed)
        self.assertIn('gpu_to_cpu', tr_2.name)
        self.assertFalse(tr_2.passed)
        self.assertIn('cpu_to_gpu', tr_3.name)
        self.assertFalse(tr_3.passed)
        self.assertIn('gpu_to_cpu_array', tr_4.name)
        self.assertFalse(tr_4.passed)
        self.assertIn('cpu_to_gpu_array', tr_5.name)
        self.assertFalse(tr_5.passed)
        self.assertIn('mass_array', tr_6.name)
        self.assertFalse(tr_6.passed)
        self.assertIn('float_array', tr_7.name)
        self.assertTrue(tr_7.passed)
        self.assertIn('float_array_diff', tr_8.name)
        self.assertFalse(tr_8.passed)
        self.assertIn('float_begin', tr_9.name)
        self.assertFalse(tr_9.passed)
        self.assertIn('float_begin_long', tr_10.name)
        self.assertFalse(tr_10.passed)
        self.assertIn('float_begin_more', tr_11.name)
        self.assertFalse(tr_11.passed)
        self.assertIn('float_begin_more_diff', tr_12.name)
        self.assertFalse(tr_12.passed)
        self.assertIn('float_begin_more_brief', tr_13.name)
        self.assertFalse(tr_13.passed)
        self.assertIn('float_begin_fail', tr_14.name)
        self.assertFalse(tr_14.passed)

    def test_run_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)
        docstrings_to_test = {'one_plus_one': "\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> # doctest: +SKIP('skip')\n                >>> print(1+1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:CPU)\n                    >>> print(1-1)\n                    0\n            ", 'one_minus_one': '\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> print(1+1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:GPU)\n                    >>> print(1-1)\n                    0\n            '}
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)
        (tr_0, tr_1, tr_2, tr_3) = test_results
        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-0', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)
        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-1', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)
        self.assertIn('one_minus_one', tr_2.name)
        self.assertIn('code-example-0', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertTrue(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertFalse(tr_2.failed)
        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-1', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertFalse(tr_3.passed)
        self.assertTrue(tr_3.skipped)
        self.assertFalse(tr_3.failed)

    def test_run_gpu(self):
        if False:
            i = 10
            return i + 15
        _clear_environ()
        test_capacity = {'cpu', 'gpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)
        docstrings_to_test = {'one_plus_one': '\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> # doctest: +REQUIRES(env: GPU)\n                >>> print(1+1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:XPU)\n                    >>> print(1-1)\n                    0\n            ', 'one_minus_one': '\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> print(1-1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:GPU, env: XPU)\n                    >>> print(1-1)\n                    0\n            '}
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)
        (tr_0, tr_1, tr_2, tr_3) = test_results
        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-0', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.skipped)
        self.assertFalse(tr_0.failed)
        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-1', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertFalse(tr_1.passed)
        self.assertTrue(tr_1.skipped)
        self.assertFalse(tr_1.failed)
        self.assertIn('one_minus_one', tr_2.name)
        self.assertIn('code-example-0', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertFalse(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertTrue(tr_2.failed)
        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-1', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertFalse(tr_3.passed)
        self.assertTrue(tr_3.skipped)
        self.assertFalse(tr_3.failed)

    def test_run_xpu_distributed(self):
        if False:
            i = 10
            return i + 15
        _clear_environ()
        test_capacity = {'cpu', 'xpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)
        docstrings_to_test = {'one_plus_one': '\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> # doctest: +REQUIRES(env: GPU)\n                >>> print(1+1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:XPU)\n                    >>> print(1-1)\n                    0\n            ', 'one_minus_one': '\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> print(1-1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:CPU, env: XPU)\n                    >>> print(1-1)\n                    0\n            '}
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)
        (tr_0, tr_1, tr_2, tr_3) = test_results
        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-0', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)
        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-1', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)
        self.assertIn('one_minus_one', tr_2.name)
        self.assertIn('code-example-0', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertFalse(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertTrue(tr_2.failed)
        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-1', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertTrue(tr_3.passed)
        self.assertFalse(tr_3.skipped)
        self.assertFalse(tr_3.failed)

    def test_style_google(self):
        if False:
            for i in range(10):
                print('nop')
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='google', target='docstring')
        doctester.prepare(test_capacity)
        docstrings_to_test = {'one_plus_one': "\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> # doctest: +SKIP('skip')\n                >>> print(1+1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:CPU)\n                    >>> print(1-1)\n                    0\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-2\n\n                    >>> print(1+2)\n                    3\n            ", 'one_minus_one': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:GPU)\n                    >>> print(1-1)\n                    0\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-2\n\n                    >>> print(1+1)\n                    3\n            '}
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)
        (tr_0, tr_1, tr_2, tr_3) = test_results
        self.assertIn('one_plus_one', tr_0.name)
        self.assertNotIn('code-example-1', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.skipped)
        self.assertFalse(tr_0.failed)
        self.assertIn('one_plus_one', tr_1.name)
        self.assertNotIn('code-example-2', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)
        self.assertIn('one_minus_one', tr_2.name)
        self.assertNotIn('code-example-1', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertFalse(tr_2.passed)
        self.assertTrue(tr_2.skipped)
        self.assertFalse(tr_2.failed)
        self.assertIn('one_minus_one', tr_3.name)
        self.assertNotIn('code-example-2', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertFalse(tr_3.passed)
        self.assertFalse(tr_3.skipped)
        self.assertTrue(tr_3.failed)
        _clear_environ()
        test_capacity = {'cpu', 'gpu'}
        doctester = Xdoctester(style='google', target='codeblock')
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)
        (tr_0, tr_1, tr_2, tr_3) = test_results
        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-1', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.skipped)
        self.assertFalse(tr_0.failed)
        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-2', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)
        self.assertIn('one_minus_one', tr_2.name)
        self.assertIn('code-example-1', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertTrue(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertFalse(tr_2.failed)
        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-2', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertFalse(tr_3.passed)
        self.assertFalse(tr_3.skipped)
        self.assertTrue(tr_3.failed)

    def test_style_freeform(self):
        if False:
            for i in range(10):
                print('nop')
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='docstring')
        doctester.prepare(test_capacity)
        docstrings_to_test = {'one_plus_one': "\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> # doctest: +SKIP('skip')\n                >>> print(1+1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:CPU)\n                    >>> print(1-1)\n                    0\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-2\n\n                    >>> print(1+2)\n                    3\n            ", 'one_minus_one': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:CPU)\n                    >>> print(1-1)\n                    0\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-2\n\n                    >>> print(1+1)\n                    3\n            '}
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)
        (tr_0, tr_1) = test_results
        self.assertIn('one_plus_one', tr_0.name)
        self.assertNotIn('code-example', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)
        self.assertIn('one_minus_one', tr_1.name)
        self.assertNotIn('code-example', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertFalse(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertTrue(tr_1.failed)
        _clear_environ()
        test_capacity = {'cpu', 'gpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)
        docstrings_to_test = {'one_plus_one': "\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> # doctest: +SKIP('skip')\n                >>> print(1+1)\n                2\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:CPU)\n                    >>> for i in range(2):\n                    ...     print(i)\n                    0\n                    1\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-2\n\n                    >>> print(1+2)\n                    3\n            ", 'one_minus_one': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +REQUIRES(env:CPU)\n                    >>> print(1-1)\n                    0\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-2\n\n                    >>> print(1+1)\n                    3\n            '}
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 5)
        (tr_0, tr_1, tr_2, tr_3, tr_4) = test_results
        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-0', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)
        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-1', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)
        self.assertIn('one_plus_one', tr_2.name)
        self.assertIn('code-example-2', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertTrue(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertFalse(tr_2.failed)
        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-1', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertTrue(tr_3.passed)
        self.assertFalse(tr_3.skipped)
        self.assertFalse(tr_3.failed)
        self.assertIn('one_minus_one', tr_4.name)
        self.assertIn('code-example-2', tr_4.name)
        self.assertFalse(tr_4.nocode)
        self.assertFalse(tr_4.passed)
        self.assertFalse(tr_4.skipped)
        self.assertTrue(tr_4.failed)

    def test_no_code(self):
        if False:
            i = 10
            return i + 15
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='google', target='docstring')
        doctester.prepare(test_capacity)
        docstrings_to_test = {'one_plus_one': "\n            placeholder\n\n            .. code-block:: python\n                :name: code-example-0\n\n                this is some blabla...\n\n                >>> # doctest: +SKIP('skip')\n                >>> print(1+1)\n                2\n            ", 'one_minus_one': '\n            placeholder\n\n            Examples:\n\n            '}
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)
        (tr_0, tr_1) = test_results
        self.assertIn('one_plus_one', tr_0.name)
        self.assertNotIn('code-example', tr_0.name)
        self.assertTrue(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertFalse(tr_0.skipped)
        self.assertFalse(tr_0.failed)
        self.assertIn('one_minus_one', tr_1.name)
        self.assertNotIn('code-example', tr_1.name)
        self.assertTrue(tr_1.nocode)
        self.assertFalse(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='google', target='codeblock')
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 0)
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='docstring')
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)
        (tr_0, tr_1) = test_results
        self.assertIn('one_plus_one', tr_0.name)
        self.assertNotIn('code-example', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)
        self.assertIn('one_minus_one', tr_1.name)
        self.assertNotIn('code-example', tr_1.name)
        self.assertTrue(tr_1.nocode)
        self.assertFalse(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 1)
        tr_0 = test_results[0]
        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)

    def test_multiprocessing_xdoctester(self):
        if False:
            return 10
        docstrings_to_test = {'static_0': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import numpy as np\n                    >>> import paddle\n                    >>> paddle.enable_static()\n                    >>> data = paddle.static.data(name='X', shape=[None, 2, 28, 28], dtype='float32')\n            ", 'static_1': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import numpy as np\n                    >>> import paddle\n                    >>> paddle.enable_static()\n                    >>> data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')\n\n            "}
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)
        (tr_0, tr_1) = test_results
        self.assertIn('static_0', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertIn('static_1', tr_1.name)
        self.assertTrue(tr_1.passed)
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(use_multiprocessing=False)
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)
        (tr_0, tr_1) = test_results
        self.assertIn('static_0', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertIn('static_1', tr_1.name)
        self.assertFalse(tr_1.passed)
        self.assertTrue(tr_1.failed)

    def test_timeout(self):
        if False:
            i = 10
            return i + 15
        docstrings_to_test = {'timeout_false': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +TIMEOUT(2)\n                    >>> import time\n                    >>> time.sleep(0.1)\n            ', 'timeout_true': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +TIMEOUT(2)\n                    >>> import time\n                    >>> time.sleep(3)\n            ', 'timeout_false_with_skip_0': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +TIMEOUT(2)\n                    >>> # doctest: +SKIP('skip')\n                    >>> import time\n                    >>> time.sleep(0.1)\n            ", 'timeout_false_with_skip_1': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +SKIP('skip')\n                    >>> # doctest: +TIMEOUT(2)\n                    >>> import time\n                    >>> time.sleep(0.1)\n            ", 'timeout_true_with_skip_0': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +TIMEOUT(2)\n                    >>> # doctest: +SKIP('skip')\n                    >>> import time\n                    >>> time.sleep(3)\n            ", 'timeout_true_with_skip_1': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +SKIP('skip')\n                    >>> # doctest: +TIMEOUT(2)\n                    >>> import time\n                    >>> time.sleep(3)\n            ", 'timeout_more_codes': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +TIMEOUT(2)\n                    >>> import time\n                    >>> time.sleep(0.1)\n\n                .. code-block:: python\n\n                    >>> # doctest: +TIMEOUT(2)\n                    >>> import time\n                    >>> time.sleep(3)\n\n            '}
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 8)
        (tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7) = test_results
        self.assertIn('timeout_false', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.timeout)
        self.assertIn('timeout_true', tr_1.name)
        self.assertFalse(tr_1.passed)
        self.assertTrue(tr_1.timeout)
        self.assertIn('timeout_false_with_skip_0', tr_2.name)
        self.assertFalse(tr_2.passed)
        self.assertFalse(tr_2.timeout)
        self.assertTrue(tr_2.skipped)
        self.assertIn('timeout_false_with_skip_1', tr_3.name)
        self.assertFalse(tr_3.passed)
        self.assertFalse(tr_3.timeout)
        self.assertTrue(tr_3.skipped)
        self.assertIn('timeout_true_with_skip_0', tr_4.name)
        self.assertFalse(tr_4.passed)
        self.assertFalse(tr_4.timeout)
        self.assertTrue(tr_4.skipped)
        self.assertIn('timeout_true_with_skip_1', tr_5.name)
        self.assertFalse(tr_5.passed)
        self.assertFalse(tr_5.timeout)
        self.assertTrue(tr_5.skipped)
        self.assertIn('timeout_more_codes', tr_6.name)
        self.assertTrue(tr_6.passed)
        self.assertFalse(tr_6.timeout)
        self.assertIn('timeout_more_codes', tr_7.name)
        self.assertFalse(tr_7.passed)
        self.assertTrue(tr_7.timeout)
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester(use_multiprocessing=False)
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 8)
        (tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7) = test_results
        self.assertIn('timeout_false', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.timeout)
        self.assertIn('timeout_true', tr_1.name)
        self.assertFalse(tr_1.passed)
        self.assertTrue(tr_1.timeout)
        self.assertIn('timeout_false_with_skip_0', tr_2.name)
        self.assertFalse(tr_2.passed)
        self.assertFalse(tr_2.timeout)
        self.assertTrue(tr_2.skipped)
        self.assertIn('timeout_false_with_skip_1', tr_3.name)
        self.assertFalse(tr_3.passed)
        self.assertFalse(tr_3.timeout)
        self.assertTrue(tr_3.skipped)
        self.assertIn('timeout_true_with_skip_0', tr_4.name)
        self.assertFalse(tr_4.passed)
        self.assertFalse(tr_4.timeout)
        self.assertTrue(tr_4.skipped)
        self.assertIn('timeout_true_with_skip_1', tr_5.name)
        self.assertFalse(tr_5.passed)
        self.assertFalse(tr_5.timeout)
        self.assertTrue(tr_5.skipped)
        self.assertIn('timeout_more_codes', tr_6.name)
        self.assertTrue(tr_6.passed)
        self.assertFalse(tr_6.timeout)
        self.assertIn('timeout_more_codes', tr_7.name)
        self.assertFalse(tr_7.passed)
        self.assertTrue(tr_7.timeout)

    def test_bad_statements(self):
        if False:
            while True:
                i = 10
        docstrings_to_test = {'bad_fluid': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import paddle.base\n            ', 'bad_fluid_from': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import paddle\n                    >>> from paddle import fluid\n            ', 'no_bad': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +SKIP('reason')\n                    >>> import os\n            ", 'bad_fluid_good_skip': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +SKIP('reason')\n                    >>> import os\n                    >>> from paddle import fluid\n            ", 'bad_fluid_bad_skip': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +SKIP('reason')\n                    >>> import os\n                    >>> from paddle import fluid\n                    >>> # doctest: +SKIP\n                    >>> import sys\n            ", 'bad_skip_mix': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +SKIP('reason')\n                    >>> import os\n                    >>> # doctest: +SKIP\n                    >>> import sys\n            ", 'bad_skip': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # doctest: +SKIP\n                    >>> import os\n\n            ', 'bad_skip_empty': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import os\n                    >>> # doctest: +SKIP()\n                    >>> import sys\n            ', 'good_skip': "\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import os\n                    >>> # doctest: +SKIP('reason')\n                    >>> import sys\n                    >>> # doctest: -SKIP\n                    >>> import math\n            ", 'comment_fluid': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> # import paddle.base\n                    >>> import os\n            ', 'oneline_skip': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import os # doctest: +SKIP\n                    >>> import sys\n            '}
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 11)
        (tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8, tr_9, tr_10) = test_results
        self.assertIn('bad_fluid', tr_0.name)
        self.assertTrue(tr_0.badstatement)
        self.assertFalse(tr_0.passed)
        self.assertIn('bad_fluid_from', tr_1.name)
        self.assertTrue(tr_1.badstatement)
        self.assertFalse(tr_1.passed)
        self.assertIn('no_bad', tr_2.name)
        self.assertFalse(tr_2.badstatement)
        self.assertFalse(tr_2.passed)
        self.assertTrue(tr_2.skipped)
        self.assertIn('bad_fluid_good_skip', tr_3.name)
        self.assertTrue(tr_3.badstatement)
        self.assertFalse(tr_3.passed)
        self.assertIn('bad_fluid_bad_skip', tr_4.name)
        self.assertTrue(tr_4.badstatement)
        self.assertFalse(tr_4.passed)
        self.assertIn('bad_skip_mix', tr_5.name)
        self.assertTrue(tr_5.badstatement)
        self.assertFalse(tr_5.passed)
        self.assertIn('bad_skip', tr_6.name)
        self.assertTrue(tr_6.badstatement)
        self.assertFalse(tr_6.passed)
        self.assertIn('bad_skip_empty', tr_7.name)
        self.assertTrue(tr_7.badstatement)
        self.assertFalse(tr_7.passed)
        self.assertIn('good_skip', tr_8.name)
        self.assertFalse(tr_8.badstatement)
        self.assertTrue(tr_8.passed)
        self.assertIn('comment_fluid', tr_9.name)
        self.assertFalse(tr_9.badstatement)
        self.assertTrue(tr_9.passed)
        self.assertIn('oneline_skip', tr_10.name)
        self.assertTrue(tr_10.badstatement)
        self.assertFalse(tr_10.passed)

    def test_bad_statements_req(self):
        if False:
            i = 10
            return i + 15
        docstrings_to_test = {'bad_required': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import sys\n                    >>> # required: GPU\n                    >>> import os\n            ', 'bad_requires': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import sys\n                    >>> # requires: GPU\n                    >>> import os\n            ', 'bad_require': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import sys\n                    >>> # require   :   GPU\n                    >>> import os\n            ', 'bad_require_2': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import sys\n                    >>> # require: GPU, xpu\n                    >>> import os\n            ', 'bad_req': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import sys\n                    >>> #require:gpu\n                    >>> import os\n            ', 'ignore_req': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import sys\n                    >>> #require:\n                    >>> import os\n            ', 'ignore_req_bad_req': '\n            this is docstring...\n\n            Examples:\n\n                .. code-block:: python\n\n                    >>> import sys\n                    >>> #require: xpu\n                    >>> import os\n                    >>> #require:\n                    >>> import os\n            '}
        _clear_environ()
        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 7)
        (tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6) = test_results
        self.assertIn('bad_required', tr_0.name)
        self.assertTrue(tr_0.badstatement)
        self.assertFalse(tr_0.passed)
        self.assertIn('bad_requires', tr_1.name)
        self.assertTrue(tr_1.badstatement)
        self.assertFalse(tr_1.passed)
        self.assertIn('bad_require', tr_2.name)
        self.assertTrue(tr_1.badstatement)
        self.assertFalse(tr_1.passed)
        self.assertIn('bad_require_2', tr_3.name)
        self.assertTrue(tr_3.badstatement)
        self.assertFalse(tr_3.passed)
        self.assertIn('bad_req', tr_4.name)
        self.assertTrue(tr_4.badstatement)
        self.assertFalse(tr_4.passed)
        self.assertIn('ignore_req', tr_5.name)
        self.assertFalse(tr_5.badstatement)
        self.assertTrue(tr_5.passed)
        self.assertIn('ignore_req_bad_req', tr_6.name)
        self.assertTrue(tr_6.badstatement)
        self.assertFalse(tr_6.passed)

    def test_single_process_directive(self):
        if False:
            i = 10
            return i + 15
        _clear_environ()
        docstrings_to_test = {'no_solo': '\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> import multiprocessing\n                    >>> p = multiprocessing.Process(\n                    ...     target=lambda a, b: a + b,\n                    ...     args=(\n                    ...     1,\n                    ...     2,\n                    ...     ),\n                    ... )\n                    >>> p.start()\n                    >>> p.join()\n            ', 'has_solo': "\n            placeholder\n\n            Examples:\n\n                .. code-block:: python\n                    :name: code-example-1\n\n                    this is some blabla...\n\n                    >>> # doctest: +SOLO('can not use add in multiprocess')\n                    >>> import multiprocessing\n                    >>> p = multiprocessing.Process(\n                    ...     target=lambda a, b: a + b,\n                    ...     args=(\n                    ...     1,\n                    ...     2,\n                    ...     ),\n                    ... )\n                    >>> p.start()\n                    >>> p.join()\n            "}
        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)
        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)
        (tr_0, tr_1) = test_results
        self.assertIn('no_solo', tr_0.name)
        self.assertFalse(tr_0.passed)
        self.assertIn('has_solo', tr_1.name)
        self.assertTrue(tr_1.passed)
if __name__ == '__main__':
    unittest.main()
import doctest
import os
import tempfile
import unittest
from apache_beam.dataframe import doctests
SAMPLE_DOCTEST = "\n>>> df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',\n...                               'Parrot', 'Parrot'],\n...                    'Max Speed': [380., 370., 24., 26.]})\n>>> df\n   Animal  Max Speed\n0  Falcon      380.0\n1  Falcon      370.0\n2  Parrot       24.0\n3  Parrot       26.0\n>>> df.groupby(['Animal']).mean()\n        Max Speed\nAnimal\nFalcon      375.0\nParrot       25.0\n"
CHECK_USES_DEFERRED_DATAFRAMES = "\n>>> type(pd).__name__\n'FakePandasObject'\n\n>>> type(pd.DataFrame([]))\n<class 'apache_beam.dataframe.frames.DeferredDataFrame'>\n\n>>> type(pd.DataFrame.from_dict({'a': [1, 2], 'b': [3, 4]}))\n<class 'apache_beam.dataframe.frames.DeferredDataFrame'>\n\n>>> pd.Index(range(10))\nRangeIndex(start=0, stop=10, step=1)\n"
WONT_IMPLEMENT_RAISING_TESTS = "\n>>> import apache_beam\n>>> raise apache_beam.dataframe.frame_base.WontImplementError('anything')\nignored exception\n>>> pd.Series(range(10)).__array__()\nignored result\n"
ERROR_RAISING_NAME_ERROR_TESTS = "\n>>> import apache_beam\n>>> raise %s('anything')\nignored exception\n>>> raise NameError\nignored exception\n>>> undefined_name\nignored exception\n>>> 2 + 2\n4\n>>> raise NameError\nfailed exception\n"
WONT_IMPLEMENT_RAISING_NAME_ERROR_TESTS = ERROR_RAISING_NAME_ERROR_TESTS % ('apache_beam.dataframe.frame_base.WontImplementError',)
NOT_IMPLEMENTED_RAISING_TESTS = "\n>>> import apache_beam\n>>> raise NotImplementedError('anything')\nignored exception\n"
NOT_IMPLEMENTED_RAISING_NAME_ERROR_TESTS = ERROR_RAISING_NAME_ERROR_TESTS % ('NotImplementedError',)
FAILED_ASSIGNMENT = "\n>>> def foo(): raise NotImplementedError()\n>>> res = 'old_value'\n>>> res = foo()\n>>> print(res)\nignored NameError\n"
RST_IPYTHON = "\nHere is an example\n.. ipython::\n\n    2 + 2\n\nsome multi-line examples\n\n.. ipython::\n\n    def foo(x):\n        return x * x\n    foo(4)\n    foo(\n        4\n    )\n\n    In [100]: def foo(x):\n       ....:     return x * x * x\n       ....:\n    foo(5)\n\nhistory is preserved\n\n    foo(3)\n    foo(4)\n\nand finally an example with pandas\n\n.. ipython::\n\n    pd.Series([1, 2, 3]).max()\n\n\nThis one should be skipped:\n\n.. ipython::\n\n   @verbatim\n   not run or tested\n\nand someting that'll fail (due to fake vs. real pandas)\n\n.. ipython::\n\n   type(pd)\n"

class DoctestTest(unittest.TestCase):

    def test_good(self):
        if False:
            i = 10
            return i + 15
        result = doctests.teststring(SAMPLE_DOCTEST, report=False)
        self.assertEqual(result.attempted, 3)
        self.assertEqual(result.failed, 0)

    def test_failure(self):
        if False:
            i = 10
            return i + 15
        result = doctests.teststring(SAMPLE_DOCTEST.replace('25.0', '25.00001'), report=False)
        self.assertEqual(result.attempted, 3)
        self.assertEqual(result.failed, 1)

    def test_uses_beam_dataframes(self):
        if False:
            for i in range(10):
                print('nop')
        result = doctests.teststring(CHECK_USES_DEFERRED_DATAFRAMES, report=False)
        self.assertNotEqual(result.attempted, 0)
        self.assertEqual(result.failed, 0)

    def test_file(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as dir:
            filename = os.path.join(dir, 'tests.py')
            with open(filename, 'w') as fout:
                fout.write(SAMPLE_DOCTEST)
            result = doctests.testfile(filename, module_relative=False, report=False)
        self.assertEqual(result.attempted, 3)
        self.assertEqual(result.failed, 0)

    def test_file_uses_beam_dataframes(self):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as dir:
            filename = os.path.join(dir, 'tests.py')
            with open(filename, 'w') as fout:
                fout.write(CHECK_USES_DEFERRED_DATAFRAMES)
            result = doctests.testfile(filename, module_relative=False, report=False)
        self.assertNotEqual(result.attempted, 0)
        self.assertEqual(result.failed, 0)

    def test_wont_implement(self):
        if False:
            print('Hello World!')
        result = doctests.teststring(WONT_IMPLEMENT_RAISING_TESTS, optionflags=doctest.ELLIPSIS, wont_implement_ok=True)
        self.assertNotEqual(result.attempted, 0)
        self.assertEqual(result.failed, 0)
        result = doctests.teststring(WONT_IMPLEMENT_RAISING_TESTS, optionflags=doctest.IGNORE_EXCEPTION_DETAIL, wont_implement_ok=True)
        self.assertNotEqual(result.attempted, 0)
        self.assertEqual(result.failed, 0)

    def test_wont_implement_followed_by_name_error(self):
        if False:
            for i in range(10):
                print('nop')
        result = doctests.teststring(WONT_IMPLEMENT_RAISING_NAME_ERROR_TESTS, optionflags=doctest.ELLIPSIS, wont_implement_ok=True)
        self.assertEqual(result.attempted, 6)
        self.assertEqual(result.failed, 1)

    def test_not_implemented(self):
        if False:
            return 10
        result = doctests.teststring(NOT_IMPLEMENTED_RAISING_TESTS, optionflags=doctest.ELLIPSIS, not_implemented_ok=True)
        self.assertNotEqual(result.attempted, 0)
        self.assertEqual(result.failed, 0)
        result = doctests.teststring(NOT_IMPLEMENTED_RAISING_TESTS, optionflags=doctest.IGNORE_EXCEPTION_DETAIL, not_implemented_ok=True)
        self.assertNotEqual(result.attempted, 0)
        self.assertEqual(result.failed, 0)

    def test_not_implemented_followed_by_name_error(self):
        if False:
            i = 10
            return i + 15
        result = doctests.teststring(NOT_IMPLEMENTED_RAISING_NAME_ERROR_TESTS, optionflags=doctest.ELLIPSIS, not_implemented_ok=True)
        self.assertEqual(result.attempted, 6)
        self.assertEqual(result.failed, 1)

    def test_failed_assignment(self):
        if False:
            while True:
                i = 10
        result = doctests.teststring(FAILED_ASSIGNMENT, optionflags=doctest.ELLIPSIS, not_implemented_ok=True)
        self.assertNotEqual(result.attempted, 0)
        self.assertEqual(result.failed, 0)

    def test_rst_ipython(self):
        if False:
            while True:
                i = 10
        try:
            import IPython
        except ImportError:
            raise unittest.SkipTest('IPython not available')
        result = doctests.test_rst_ipython(RST_IPYTHON, 'test_rst_ipython')
        self.assertEqual(result.attempted, 8)
        self.assertEqual(result.failed, 1)
if __name__ == '__main__':
    unittest.main()
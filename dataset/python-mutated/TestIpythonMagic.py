"""Tests for the Cython magics extension."""
from __future__ import absolute_import
import os
import io
import sys
from contextlib import contextmanager
from unittest import skipIf
from Cython.Build import IpythonMagic
from Cython.TestUtils import CythonTest
from Cython.Compiler.Annotate import AnnotationCCodeWriter
try:
    import IPython.testing.globalipapp
except ImportError:

    def skip_if_not_installed(_):
        if False:
            i = 10
            return i + 15
        return None
else:

    def skip_if_not_installed(c):
        if False:
            while True:
                i = 10
        return c
skip_win32 = skipIf(sys.platform == 'win32', 'Skip on Windows')
skip_py27 = skipIf(sys.version_info[:2] == (2, 7), 'Disabled in Py2.7')
try:
    from IPython.core.history import HistoryManager
    HistoryManager.enabled = False
except ImportError:
    pass

@contextmanager
def capture_output():
    if False:
        i = 10
        return i + 15
    backup = (sys.stdout, sys.stderr)
    try:
        replacement = [io.TextIOWrapper(io.BytesIO(), encoding=sys.stdout.encoding), io.TextIOWrapper(io.BytesIO(), encoding=sys.stderr.encoding)]
        (sys.stdout, sys.stderr) = replacement
        output = []
        yield output
    finally:
        (sys.stdout, sys.stderr) = backup
        for wrapper in replacement:
            wrapper.seek(0)
            output.append(wrapper.read())
            wrapper.close()
code = u'def f(x):\n    return 2*x\n'
cython3_code = u'def f(int x):\n    return 2 / x\n\ndef call(x):\n    return f(*(x,))\n'
pgo_cython3_code = cython3_code + u'def main():\n    for _ in range(100): call(5)\nmain()\n'
compile_error_code = u'cdef extern from *:\n    """\n    xxx a=1;\n    """\n    int a;\ndef doit():\n    return a\n'
compile_warning_code = u'cdef extern from *:\n    """\n    #pragma message ( "CWarning" )\n    int a = 42;\n    """\n    int a;\ndef doit():\n    return a\n'

@skip_if_not_installed
class TestIPythonMagic(CythonTest):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        CythonTest.setUpClass()
        cls._ip = IPython.testing.globalipapp.get_ipython()

    def setUp(self):
        if False:
            return 10
        CythonTest.setUp(self)
        self._ip.extension_manager.load_extension('cython')

    def test_cython_inline(self):
        if False:
            return 10
        ip = self._ip
        ip.ex('a=10; b=20')
        result = ip.run_cell_magic('cython_inline', '', 'return a+b')
        self.assertEqual(result, 30)

    @skip_win32
    def test_cython_pyximport(self):
        if False:
            i = 10
            return i + 15
        ip = self._ip
        module_name = '_test_cython_pyximport'
        ip.run_cell_magic('cython_pyximport', module_name, code)
        ip.ex('g = f(10)')
        self.assertEqual(ip.user_ns['g'], 20.0)
        ip.run_cell_magic('cython_pyximport', module_name, code)
        ip.ex('h = f(-10)')
        self.assertEqual(ip.user_ns['h'], -20.0)
        try:
            os.remove(module_name + '.pyx')
        except OSError:
            pass

    def test_cython(self):
        if False:
            for i in range(10):
                print('nop')
        ip = self._ip
        ip.run_cell_magic('cython', '', code)
        ip.ex('g = f(10)')
        self.assertEqual(ip.user_ns['g'], 20.0)

    def test_cython_name(self):
        if False:
            print('Hello World!')
        ip = self._ip
        ip.run_cell_magic('cython', '--name=mymodule', code)
        ip.ex('import mymodule; g = mymodule.f(10)')
        self.assertEqual(ip.user_ns['g'], 20.0)

    def test_cython_language_level(self):
        if False:
            return 10
        ip = self._ip
        ip.run_cell_magic('cython', '', cython3_code)
        ip.ex('g = f(10); h = call(10)')
        if sys.version_info[0] < 3:
            self.assertEqual(ip.user_ns['g'], 2 // 10)
            self.assertEqual(ip.user_ns['h'], 2 // 10)
        else:
            self.assertEqual(ip.user_ns['g'], 2.0 / 10.0)
            self.assertEqual(ip.user_ns['h'], 2.0 / 10.0)

    def test_cython3(self):
        if False:
            print('Hello World!')
        ip = self._ip
        ip.run_cell_magic('cython', '-3', cython3_code)
        ip.ex('g = f(10); h = call(10)')
        self.assertEqual(ip.user_ns['g'], 2.0 / 10.0)
        self.assertEqual(ip.user_ns['h'], 2.0 / 10.0)

    def test_cython2(self):
        if False:
            for i in range(10):
                print('nop')
        ip = self._ip
        ip.run_cell_magic('cython', '-2', cython3_code)
        ip.ex('g = f(10); h = call(10)')
        self.assertEqual(ip.user_ns['g'], 2 // 10)
        self.assertEqual(ip.user_ns['h'], 2 // 10)

    def test_cython_compile_error_shown(self):
        if False:
            return 10
        ip = self._ip
        with capture_output() as out:
            ip.run_cell_magic('cython', '-3', compile_error_code)
        (captured_out, captured_err) = out
        captured_all = captured_out + '\n' + captured_err
        self.assertTrue('error' in captured_all, msg='error in ' + captured_all)

    def test_cython_link_error_shown(self):
        if False:
            while True:
                i = 10
        ip = self._ip
        with capture_output() as out:
            ip.run_cell_magic('cython', '-3 -l=xxxxxxxx', code)
        (captured_out, captured_err) = out
        captured_all = captured_out + '\n!' + captured_err
        self.assertTrue('error' in captured_all, msg='error in ' + captured_all)

    def test_cython_warning_shown(self):
        if False:
            i = 10
            return i + 15
        ip = self._ip
        with capture_output() as out:
            ip.run_cell_magic('cython', '-3 -f', compile_warning_code)
        (captured_out, captured_err) = out
        self.assertTrue('CWarning' in captured_out)

    @skip_py27
    @skip_win32
    def test_cython3_pgo(self):
        if False:
            i = 10
            return i + 15
        ip = self._ip
        ip.run_cell_magic('cython', '-3 --pgo', pgo_cython3_code)
        ip.ex('g = f(10); h = call(10); main()')
        self.assertEqual(ip.user_ns['g'], 2.0 / 10.0)
        self.assertEqual(ip.user_ns['h'], 2.0 / 10.0)

    @skip_win32
    def test_extlibs(self):
        if False:
            for i in range(10):
                print('nop')
        ip = self._ip
        code = u'\nfrom libc.math cimport sin\nx = sin(0.0)\n        '
        ip.user_ns['x'] = 1
        ip.run_cell_magic('cython', '-l m', code)
        self.assertEqual(ip.user_ns['x'], 0)

    def test_cython_verbose(self):
        if False:
            for i in range(10):
                print('nop')
        ip = self._ip
        ip.run_cell_magic('cython', '--verbose', code)
        ip.ex('g = f(10)')
        self.assertEqual(ip.user_ns['g'], 20.0)

    def test_cython_verbose_thresholds(self):
        if False:
            return 10

        @contextmanager
        def mock_distutils():
            if False:
                return 10

            class MockLog:
                DEBUG = 1
                INFO = 2
                thresholds = [INFO]

                def set_threshold(self, val):
                    if False:
                        print('Hello World!')
                    self.thresholds.append(val)
                    return self.thresholds[-2]
            new_log = MockLog()
            old_log = IpythonMagic.distutils.log
            try:
                IpythonMagic.distutils.log = new_log
                yield new_log
            finally:
                IpythonMagic.distutils.log = old_log
        ip = self._ip
        with mock_distutils() as verbose_log:
            ip.run_cell_magic('cython', '--verbose', code)
            ip.ex('g = f(10)')
        self.assertEqual(ip.user_ns['g'], 20.0)
        self.assertEqual([verbose_log.INFO, verbose_log.DEBUG, verbose_log.INFO], verbose_log.thresholds)
        with mock_distutils() as normal_log:
            ip.run_cell_magic('cython', '', code)
            ip.ex('g = f(10)')
        self.assertEqual(ip.user_ns['g'], 20.0)
        self.assertEqual([normal_log.INFO], normal_log.thresholds)

    def test_cython_no_annotate(self):
        if False:
            return 10
        ip = self._ip
        html = ip.run_cell_magic('cython', '', code)
        self.assertTrue(html is None)

    def test_cython_annotate(self):
        if False:
            while True:
                i = 10
        ip = self._ip
        html = ip.run_cell_magic('cython', '--annotate', code)
        self.assertTrue(AnnotationCCodeWriter.COMPLETE_CODE_TITLE not in html.data)

    def test_cython_annotate_default(self):
        if False:
            while True:
                i = 10
        ip = self._ip
        html = ip.run_cell_magic('cython', '-a', code)
        self.assertTrue(AnnotationCCodeWriter.COMPLETE_CODE_TITLE not in html.data)

    def test_cython_annotate_complete_c_code(self):
        if False:
            for i in range(10):
                print('nop')
        ip = self._ip
        html = ip.run_cell_magic('cython', '--annotate-fullc', code)
        self.assertTrue(AnnotationCCodeWriter.COMPLETE_CODE_TITLE in html.data)
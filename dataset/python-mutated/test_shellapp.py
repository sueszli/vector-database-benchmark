"""Tests for shellapp module.

Authors
-------
* Bradley Froehle
"""
import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt

class TestFileToRun(tt.TempFileMixin, unittest.TestCase):
    """Test the behavior of the file_to_run parameter."""

    def test_py_script_file_attribute(self):
        if False:
            print('Hello World!')
        'Test that `__file__` is set when running `ipython file.py`'
        src = 'print(__file__)\n'
        self.mktmp(src)
        err = None
        tt.ipexec_validate(self.fname, self.fname, err)

    def test_ipy_script_file_attribute(self):
        if False:
            while True:
                i = 10
        'Test that `__file__` is set when running `ipython file.ipy`'
        src = 'print(__file__)\n'
        self.mktmp(src, ext='.ipy')
        err = None
        tt.ipexec_validate(self.fname, self.fname, err)

    @dec.skip_win32
    def test_py_script_file_attribute_interactively(self):
        if False:
            return 10
        'Test that `__file__` is not set after `ipython -i file.py`'
        src = 'True\n'
        self.mktmp(src)
        (out, err) = tt.ipexec(self.fname, options=['-i'], commands=['"__file__" in globals()', 'print(123)', 'exit()'])
        assert 'False' in out, f'Subprocess stderr:\n{err}\n-----'
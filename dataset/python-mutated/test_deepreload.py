"""Test suite for the deepreload module."""
import types
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
from IPython.lib.deepreload import modules_reloading
from IPython.lib.deepreload import reload as dreload
from IPython.utils.syspathcontext import prepended_to_syspath

def test_deepreload():
    if False:
        i = 10
        return i + 15
    'Test that dreload does deep reloads and skips excluded modules.'
    with TemporaryDirectory() as tmpdir:
        with prepended_to_syspath(tmpdir):
            tmpdirpath = Path(tmpdir)
            with open(tmpdirpath / 'A.py', 'w', encoding='utf-8') as f:
                f.write('class Object:\n    pass\nok = True\n')
            with open(tmpdirpath / 'B.py', 'w', encoding='utf-8') as f:
                f.write("import A\nassert A.ok, 'we are fine'\n")
            import A
            import B
            obj = A.Object()
            dreload(B, exclude=['A'])
            assert isinstance(obj, A.Object) is True
            A.ok = False
            with pytest.raises(AssertionError, match='we are fine'):
                dreload(B, exclude=['A'])
            assert len(modules_reloading) == 0
            assert not A.ok
            obj = A.Object()
            A.ok = False
            dreload(B)
            assert A.ok
            assert isinstance(obj, A.Object) is False

def test_not_module():
    if False:
        while True:
            i = 10
    pytest.raises(TypeError, dreload, 'modulename')

def test_not_in_sys_modules():
    if False:
        i = 10
        return i + 15
    fake_module = types.ModuleType('fake_module')
    with pytest.raises(ImportError, match='not in sys.modules'):
        dreload(fake_module)
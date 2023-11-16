"""Basic tests over Python 3 targets."""
from pytype.tests import test_base

class TestExec(test_base.BaseTest):
    """Basic tests."""

    def test_exec_function(self):
        if False:
            print('Hello World!')
        self.assertNoCrash(self.Check, '\n      g = {}\n      exec("a = 11", g, g)\n      assert g[\'a\'] == 11\n      ')

    def test_import_shadowed(self):
        if False:
            print('Hello World!')
        'Test that we import modules from pytd/ rather than typeshed.'
        for module in ['importlib', 're', 'signal']:
            ty = self.Infer(f'import {module}')
            self.assertTypesMatchPytd(ty, f'import {module}')

    def test_cleanup(self):
        if False:
            return 10
        ty = self.Infer('\n      with open("foo.py", "r") as f:\n        v = f.read()\n      w = 42\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TextIO\n      f = ...  # type: TextIO\n      v = ...  # type: str\n      w = ...  # type: int\n    ')
if __name__ == '__main__':
    test_base.main()
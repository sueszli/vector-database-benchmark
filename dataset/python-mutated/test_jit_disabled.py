import sys
import os
import contextlib
import subprocess
from torch.testing._internal.common_utils import TestCase, run_tests, TemporaryFileName

@contextlib.contextmanager
def _jit_disabled():
    if False:
        i = 10
        return i + 15
    cur_env = os.environ.get('PYTORCH_JIT', '1')
    os.environ['PYTORCH_JIT'] = '0'
    try:
        yield
    finally:
        os.environ['PYTORCH_JIT'] = cur_env

class TestJitDisabled(TestCase):
    """
    These tests are separate from the rest of the JIT tests because we need
    run a new subprocess and `import torch` with the correct environment
    variables set.
    """

    def compare_enabled_disabled(self, src):
        if False:
            return 10
        '\n        Runs the script in `src` with PYTORCH_JIT enabled and disabled and\n        compares their stdout for equality.\n        '
        with TemporaryFileName() as fname:
            with open(fname, 'w') as f:
                f.write(src)
                with _jit_disabled():
                    out_disabled = subprocess.check_output([sys.executable, fname])
                out_enabled = subprocess.check_output([sys.executable, fname])
                self.assertEqual(out_disabled, out_enabled)

    def test_attribute(self):
        if False:
            i = 10
            return i + 15
        _program_string = '\nimport torch\n\nclass Foo(torch.jit.ScriptModule):\n    def __init__(self, x):\n        super().__init__()\n        self.x = torch.jit.Attribute(x, torch.Tensor)\n\n    def forward(self, input):\n        return input\n\ns = Foo(torch.ones(2, 3))\nprint(s.x)\n'
        self.compare_enabled_disabled(_program_string)

    def test_script_module_construction(self):
        if False:
            return 10
        _program_string = '\nimport torch\n\nclass AModule(torch.jit.ScriptModule):\n    @torch.jit.script_method\n    def forward(self, input):\n        pass\n\nAModule()\nprint("Didn\'t throw exception")\n'
        self.compare_enabled_disabled(_program_string)

    def test_recursive_script(self):
        if False:
            for i in range(10):
                print('nop')
        _program_string = '\nimport torch\n\nclass AModule(torch.nn.Module):\n    def forward(self, input):\n        pass\n\nsm = torch.jit.script(AModule())\nprint("Didn\'t throw exception")\n'
        self.compare_enabled_disabled(_program_string)
if __name__ == '__main__':
    run_tests()
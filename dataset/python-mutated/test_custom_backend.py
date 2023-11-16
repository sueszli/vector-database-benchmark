import os
import tempfile
import torch
from backend import Model, to_custom_backend, get_custom_backend_library_path
from torch.testing._internal.common_utils import TestCase, run_tests

class TestCustomBackend(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.library_path = get_custom_backend_library_path()
        torch.ops.load_library(self.library_path)
        self.model = to_custom_backend(torch.jit.script(Model()))

    def test_execute(self):
        if False:
            return 10
        '\n        Test execution using the custom backend.\n        '
        a = torch.randn(4)
        b = torch.randn(4)
        expected = (a + b, a - b)
        out = self.model(a, b)
        self.assertTrue(expected[0].allclose(out[0]))
        self.assertTrue(expected[1].allclose(out[1]))

    def test_save_load(self):
        if False:
            return 10
        '\n        Test that a lowered module can be executed correctly\n        after saving and loading.\n        '
        self.test_execute()
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.close()
            torch.jit.save(self.model, f.name)
            loaded = torch.jit.load(f.name)
        finally:
            os.unlink(f.name)
        self.model = loaded
        self.test_execute()
if __name__ == '__main__':
    run_tests()
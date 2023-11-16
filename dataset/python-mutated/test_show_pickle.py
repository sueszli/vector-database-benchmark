import unittest
import io
import tempfile
import torch
import torch.utils.show_pickle
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS

class TestShowPickle(TestCase):

    @unittest.skipIf(IS_WINDOWS, "Can't re-open temp file on Windows")
    def test_scripted_model(self):
        if False:
            return 10

        class MyCoolModule(torch.nn.Module):

            def __init__(self, weight):
                if False:
                    print('Hello World!')
                super().__init__()
                self.weight = weight

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x * self.weight
        m = torch.jit.script(MyCoolModule(torch.tensor([2.0])))
        with tempfile.NamedTemporaryFile() as tmp:
            torch.jit.save(m, tmp)
            tmp.flush()
            buf = io.StringIO()
            torch.utils.show_pickle.main(['', tmp.name + '@*/data.pkl'], output_stream=buf)
            output = buf.getvalue()
            self.assertRegex(output, 'MyCoolModule')
            self.assertRegex(output, 'weight')
if __name__ == '__main__':
    run_tests()
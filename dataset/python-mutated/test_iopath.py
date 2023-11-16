import unittest
from unittest import mock

class TestIOPath(unittest.TestCase):

    def test_no_iopath(self):
        if False:
            print('Hello World!')
        from .test_reproducibility import TestReproducibility
        with mock.patch.dict('sys.modules', {'iopath': None}):
            TestReproducibility._test_reproducibility(self, 'test_reproducibility')

    def test_no_supports_rename(self):
        if False:
            return 10
        from .test_reproducibility import TestReproducibility
        with mock.patch('fairseq.file_io.PathManager.supports_rename') as mock_fn:
            mock_fn.return_value = False
            TestReproducibility._test_reproducibility(self, 'test_reproducibility')
if __name__ == '__main__':
    unittest.main()
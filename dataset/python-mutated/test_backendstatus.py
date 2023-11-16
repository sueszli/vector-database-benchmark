"""
Test the BackendStatus.
"""
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeLondon
from qiskit.providers.models import BackendStatus

class TestBackendConfiguration(QiskitTestCase):
    """Test the BackendStatus class."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Test backend status for one of the fake backends'
        super().setUp()
        self.backend_status = BackendStatus('my_backend', '1.0', True, 2, 'online')

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        'Test representation methods of BackendStatus'
        self.assertIsInstance(self.backend_status.__repr__(), str)
        repr_html = self.backend_status._repr_html_()
        self.assertIsInstance(repr_html, str)
        self.assertIn(self.backend_status.backend_name, repr_html)

    def test_fake_backend_status(self):
        if False:
            print('Hello World!')
        'Test backend status for one of the fake backends'
        fake_backend = FakeLondon()
        backend_status = fake_backend.status()
        self.assertIsInstance(backend_status, BackendStatus)
if __name__ == '__main__':
    import unittest
    unittest.main()
"""Test for deprecation of qiskit.test.mock module."""
from qiskit.test import QiskitTestCase

class MockModuleDeprecationTest(QiskitTestCase):
    """Test for deprecation of qiskit.test.mock module."""

    def test_deprecated_mock_module(self):
        if False:
            i = 10
            return i + 15
        'Test that the mock module is deprecated.'
        with self.assertWarns(DeprecationWarning):
            from qiskit.test.mock import FakeWashington
        with self.assertWarns(DeprecationWarning):
            from qiskit.test.mock.backends import FakeWashington
        with self.assertWarns(DeprecationWarning):
            from qiskit.test.mock.backends.washington import FakeWashington
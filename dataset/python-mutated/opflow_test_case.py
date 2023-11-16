"""Opflow Test Case"""
import warnings
from qiskit.test import QiskitTestCase

class QiskitOpflowTestCase(QiskitTestCase):
    """Opflow test Case"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*opflow.*')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        warnings.filterwarnings('error', category=DeprecationWarning, message='.*opflow.*')
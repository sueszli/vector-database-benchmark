"""Tests for the wrapper functionality."""
import sys
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from io import StringIO
import qiskit
from qiskit import providers
from qiskit.tools.monitor import backend_overview, backend_monitor
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeProviderFactory, FakeBackend, FakeVigo

class TestBackendOverview(QiskitTestCase):
    """Tools test case."""

    def _restore_ibmq(self):
        if False:
            i = 10
            return i + 15
        if not self.import_error:
            qiskit.IBMQ = self.ibmq_back
        else:
            del qiskit.IBMQ
        if self.prov_backup:
            providers.ibmq = self.prov_backup
        else:
            del providers.ibmq

    def _restore_ibmq_mod(self):
        if False:
            i = 10
            return i + 15
        if self.ibmq_module_backup is not None:
            sys.modules['qiskit.providers.ibmq'] = self.ibmq_module_backup
        else:
            sys.modules.pop('qiskit.providers.ibmq')

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        ibmq_mock = MagicMock()
        ibmq_mock.IBMQBackend = FakeBackend
        if 'qiskit.providers.ibmq' in sys.modules:
            self.ibmq_module_backup = sys.modules['qiskit.providers.ibmq']
        else:
            self.ibmq_module_backup = None
        sys.modules['qiskit.providers.ibmq'] = ibmq_mock
        self.addCleanup(self._restore_ibmq_mod)
        if hasattr(qiskit, 'IBMQ'):
            self.import_error = False
        else:
            self.import_error = True
            qiskit.IBMQ = None
        self.ibmq_back = qiskit.IBMQ
        qiskit.IBMQ = FakeProviderFactory()
        self.addCleanup(self._restore_ibmq)
        if hasattr(providers, 'ibmq'):
            self.prov_backup = providers.ibmq
        else:
            self.prov_backup = None
        providers.ibmq = MagicMock()

    @patch('qiskit.tools.monitor.overview.get_unique_backends', return_value=[FakeVigo()])
    def test_backend_overview(self, _):
        if False:
            for i in range(10):
                print('nop')
        'Test backend_overview'
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            backend_overview()
        stdout = fake_stdout.getvalue()
        self.assertIn('Operational:', stdout)
        self.assertIn('Avg. T1:', stdout)
        self.assertIn('Num. Qubits:', stdout)

    @patch('qiskit.tools.monitor.overview.get_unique_backends', return_value=[FakeVigo()])
    def test_backend_monitor(self, _):
        if False:
            print('Hello World!')
        'Test backend_monitor'
        for back in [FakeVigo()]:
            if not back.configuration().simulator:
                backend = back
                break
        with patch('sys.stdout', new=StringIO()) as fake_stdout:
            backend_monitor(backend)
        stdout = fake_stdout.getvalue()
        self.assertIn('Configuration', stdout)
        self.assertIn('Qubits [Name / Freq / T1 / T2 / ', stdout)
        for gate in backend.properties().gates:
            if gate.gate not in ['id'] and len(gate.qubits) == 1:
                self.assertIn(gate.gate.upper() + ' err', stdout)
        self.assertIn('Readout err', stdout)
        self.assertIn('Multi-Qubit Gates [Name / Type / Gate Error]', stdout)
if __name__ == '__main__':
    unittest.main(verbosity=2)
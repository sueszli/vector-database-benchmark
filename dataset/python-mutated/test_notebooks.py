"""Tests for the wrapper functionality."""
import os
import sys
import unittest
from qiskit.utils import optionals
from qiskit.test import Path, QiskitTestCase, slow_test
TIMEOUT = 1000
JUPYTER_KERNEL = 'python3'

@unittest.skipUnless(optionals.HAS_IBMQ, 'requires IBMQ provider')
@unittest.skipUnless(optionals.HAS_JUPYTER, 'involves running Jupyter notebooks')
class TestJupyter(QiskitTestCase):
    """Notebooks test case."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.execution_path = os.path.join(Path.SDK.value, '..')
        self.notebook_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'notebooks')

    def _execute_notebook(self, filename):
        if False:
            while True:
                i = 10
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        execute_preprocessor = ExecutePreprocessor(timeout=TIMEOUT, kernel_name=JUPYTER_KERNEL)
        with open(filename) as file_:
            notebook = nbformat.read(file_, as_version=4)
        top_str = "\n        import qiskit\n        import qiskit.providers.ibmq\n        import sys\n        from unittest.mock import create_autospec, MagicMock\n        from qiskit.providers.fake_provider import FakeProviderFactory\n        from qiskit.providers import basicaer\n        fake_prov = FakeProviderFactory()\n        qiskit.IBMQ = fake_prov\n        ibmq_mock = create_autospec(basicaer)\n        ibmq_mock.IBMQJobApiError = MagicMock()\n        sys.modules['qiskit.providers.ibmq'] = ibmq_mock\n        sys.modules['qiskit.providers.ibmq.job'] = ibmq_mock\n        sys.modules['qiskit.providers.ibmq.job.exceptions'] = ibmq_mock\n        "
        top = nbformat.notebooknode.NotebookNode({'cell_type': 'code', 'execution_count': 0, 'metadata': {}, 'outputs': [], 'source': top_str})
        notebook.cells = [top] + notebook.cells
        execute_preprocessor.preprocess(notebook, {'metadata': {'path': self.execution_path}})

    @unittest.skipIf(sys.platform != 'linux', 'Fails with Python >=3.8 on osx and windows')
    def test_jupyter_jobs_pbars(self):
        if False:
            return 10
        'Test Jupyter progress bars and job status functionality'
        self._execute_notebook(os.path.join(self.notebook_dir, 'test_pbar_status.ipynb'))

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, 'matplotlib not available.')
    @slow_test
    def test_backend_tools(self):
        if False:
            while True:
                i = 10
        'Test Jupyter backend tools.'
        self._execute_notebook(os.path.join(self.notebook_dir, 'test_backend_tools.ipynb'))
if __name__ == '__main__':
    unittest.main(verbosity=2)
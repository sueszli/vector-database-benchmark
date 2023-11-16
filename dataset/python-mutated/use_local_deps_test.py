import os.path
import sys
import internal_unit_testing
import pytest

@pytest.fixture(scope='module', autouse=True)
def local_deps():
    if False:
        for i in range(10):
            print('nop')
    'Add local directory to the PYTHONPATH to allow absolute imports.\n\n    Relative imports do not work in Airflow workflow definitions.\n    '
    workflows_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(workflows_dir)
    yield
    sys.path.remove(workflows_dir)

def test_dag_import():
    if False:
        for i in range(10):
            print('nop')
    'Test that the DAG file can be successfully imported.\n\n    This tests that the DAG can be parsed, but does not run it in an Airflow\n    environment. This is a recommended confidence check by the official Airflow\n    docs: https://airflow.incubator.apache.org/tutorial.html#testing\n    '
    from . import use_local_deps as module
    internal_unit_testing.assert_has_valid_dag(module)
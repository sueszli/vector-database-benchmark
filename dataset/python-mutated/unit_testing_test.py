from airflow import exceptions
from airflow import models
import internal_unit_testing
import pytest

@pytest.fixture(scope='function')
def set_variables(airflow_database):
    if False:
        return 10
    models.Variable.set('gcp_project', 'example-project')
    yield
    models.Variable.delete('gcp_project')

def test_dag_no_dag():
    if False:
        i = 10
        return i + 15
    import internal_unit_testing as module
    with pytest.raises(AssertionError):
        internal_unit_testing.assert_has_valid_dag(module)

def test_dag_has_cycle():
    if False:
        i = 10
        return i + 15
    from . import unit_testing_cycle as module
    with pytest.raises(exceptions.AirflowDagCycleException):
        internal_unit_testing.assert_has_valid_dag(module)

def test_dag_with_variables(set_variables):
    if False:
        return 10
    from . import unit_testing_variables as module
    internal_unit_testing.assert_has_valid_dag(module)
import internal_unit_testing

def test_dag_import(airflow_database):
    if False:
        for i in range(10):
            print('nop')
    'Test that the DAG file can be successfully imported.\n\n    This tests that the DAG can be parsed, but does not run it in an Airflow\n    environment. This is a recommended confidence check by the official Airflow\n    docs: https://airflow.incubator.apache.org/tutorial.html#testing\n    '
    from . import airflow_db_cleanup as module
    internal_unit_testing.assert_has_valid_dag(module)
import internal_unit_testing

def test_dag_import():
    if False:
        print('Hello World!')
    'Test that the DAG file can be successfully imported.\n    This tests that the DAG can be parsed, but does not run it in an Airflow\n    environment. This is a recommended confidence check by the official Airflow\n    docs: https://airflow.incubator.apache.org/tutorial.html#testing\n    '
    from . import kubernetes_pod_operator_c2 as module
    internal_unit_testing.assert_has_valid_dag(module)
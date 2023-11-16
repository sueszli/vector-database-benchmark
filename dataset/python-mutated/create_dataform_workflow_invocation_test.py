import internal_unit_testing

def test_create_dataform_workflow_invocation() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Test that the DAG file can be successfully imported.\n    This tests that the DAG can be parsed, but does not run it in an Airflow\n    environment. This is a recommended confidence check by the official Airflow\n    docs: https://airflow.incubator.apache.org/tutorial.html#testing\n    '
    from . import create_dataform_workflow_invocation
    internal_unit_testing.assert_has_valid_dag(create_dataform_workflow_invocation)
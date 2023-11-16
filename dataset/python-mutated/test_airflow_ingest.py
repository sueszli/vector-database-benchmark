from with_airflow.repository import airflow_complex_dag, airflow_simple_dag, airflow_simple_dag_with_execution_date

def test_airflow_simple_dag():
    if False:
        for i in range(10):
            print('nop')
    assert airflow_simple_dag.execute_in_process()

def test_airflow_complex_dag():
    if False:
        i = 10
        return i + 15
    assert airflow_complex_dag.execute_in_process()

def test_airflow_simple_dag_with_execution_date():
    if False:
        while True:
            i = 10
    assert airflow_simple_dag_with_execution_date.execute_in_process()
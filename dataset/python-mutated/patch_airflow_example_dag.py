from dagster_airflow.utils import is_airflow_2_loaded_in_environment

def patch_airflow_example_dag(dag_bag):
    if False:
        i = 10
        return i + 15
    if is_airflow_2_loaded_in_environment():
        return
    dag = dag_bag.dags.get('example_complex')
    task = dag.get_task('search_catalog')
    task.op_args = ['dummy']
    return
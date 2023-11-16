import logging
from dagster_airflow.operators.util import invoke_steps_within_python_operator
from dagster_airflow.vendor.python_operator import PythonOperator

class CustomOperator(PythonOperator):

    def __init__(self, dagster_operator_parameters, *args, **kwargs):
        if False:
            while True:
                i = 10

        def python_callable(ts, dag_run, **kwargs):
            if False:
                i = 10
                return i + 15
            logger = logging.getLogger('CustomOperatorLogger')
            logger.setLevel(logging.INFO)
            logger.info('CustomOperator is called')
            return invoke_steps_within_python_operator(dagster_operator_parameters.invocation_args, ts, dag_run, **kwargs)
        super(CustomOperator, self).__init__(*args, task_id=dagster_operator_parameters.task_id, provide_context=True, python_callable=python_callable, dag=dagster_operator_parameters.dag, **kwargs)
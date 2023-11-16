from __future__ import annotations
from typing import TYPE_CHECKING, Any
import httpx
import pendulum
from airflow.decorators import dag, task
from airflow.models.baseoperator import BaseOperator
from airflow.operators.email import EmailOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class GetRequestOperator(BaseOperator):
    """Custom operator to send GET request to provided url"""

    def __init__(self, *, url: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.url = url

    def execute(self, context: Context):
        if False:
            print('Hello World!')
        return httpx.get(self.url).json()

@dag(schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example'])
def example_dag_decorator(email: str='example@example.com'):
    if False:
        i = 10
        return i + 15
    '\n    DAG to send server IP to email.\n\n    :param email: Email to send IP to. Defaults to example@example.com.\n    '
    get_ip = GetRequestOperator(task_id='get_ip', url='http://httpbin.org/get')

    @task(multiple_outputs=True)
    def prepare_email(raw_json: dict[str, Any]) -> dict[str, str]:
        if False:
            while True:
                i = 10
        external_ip = raw_json['origin']
        return {'subject': f'Server connected from {external_ip}', 'body': f'Seems like today your server executing Airflow is connected from IP {external_ip}<br>'}
    email_info = prepare_email(get_ip.output)
    EmailOperator(task_id='send_email', to=email, subject=email_info['subject'], html_content=email_info['body'])
example_dag = example_dag_decorator()
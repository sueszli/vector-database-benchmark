from __future__ import annotations
import os
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.providers.zendesk.hooks.zendesk import ZendeskHook
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'zendesk_custom_get_dag'

@task
def fetch_organizations() -> list[dict]:
    if False:
        for i in range(10):
            print('nop')
    hook = ZendeskHook()
    response = hook.get(url='https://yourdomain.zendesk.com/api/v2/organizations.json')
    return [org.to_dict() for org in response]
with DAG(dag_id=DAG_ID, schedule=None, start_date=datetime(2021, 1, 1), catchup=False) as dag:
    fetch_organizations()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)
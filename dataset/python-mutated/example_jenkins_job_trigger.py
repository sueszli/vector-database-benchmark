from __future__ import annotations
import os
from datetime import datetime
from requests import Request
from airflow import DAG
from airflow.decorators import task
from airflow.providers.jenkins.hooks.jenkins import JenkinsHook
from airflow.providers.jenkins.operators.jenkins_job_trigger import JenkinsJobTriggerOperator
JENKINS_CONNECTION_ID = 'your_jenkins_connection'
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'test_jenkins'
with DAG(DAG_ID, default_args={'retries': 1, 'concurrency': 8, 'max_active_runs': 8}, start_date=datetime(2017, 6, 1), schedule=None) as dag:
    job_trigger = JenkinsJobTriggerOperator(task_id='trigger_job', job_name='generate-merlin-config', parameters={'first_parameter': 'a_value', 'second_parameter': '18'}, jenkins_connection_id=JENKINS_CONNECTION_ID)

    @task
    def grab_artifact_from_jenkins(url):
        if False:
            i = 10
            return i + 15
        "\n        Grab an artifact from the previous job\n        The python-jenkins library doesn't expose a method for that\n        But it's totally possible to build manually the request for that\n        "
        hook = JenkinsHook(JENKINS_CONNECTION_ID)
        jenkins_server = hook.get_jenkins_server()
        url += 'artifact/myartifact.xml'
        request = Request(method='GET', url=url)
        response = jenkins_server.jenkins_open(request)
        return response
    grab_artifact_from_jenkins(job_trigger.output)
from tests.system.utils import get_test_run
test_run = get_test_run(dag)
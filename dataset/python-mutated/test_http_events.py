from time import sleep
import pytest
import ray
from ray import workflow
from ray.workflow.http_event_provider import HTTPListener
from ray.tests.conftest import *
from ray import serve
from ray.workflow import common
from ray._private.test_utils import wait_for_condition
import requests

@pytest.mark.parametrize('workflow_start_regular_shared_serve', [{'num_cpus': 4}], indirect=True)
def test_receive_event_by_http(workflow_start_regular_shared_serve):
    if False:
        i = 10
        return i + 15
    'This test has a statically declared event workflow task,\n    receiving one externally posted message to a Ray Serve endpoint.\n    '

    def send_event():
        if False:
            return 10
        resp = requests.post('http://127.0.0.1:8000/event/send_event/' + 'workflow_test_receive_event_by_http', json={'event_key': 'event_key', 'event_payload': 'event_message'})
        return resp
    event_promise = workflow.wait_for_event(HTTPListener, event_key='event_key')
    workflow.run_async(event_promise, workflow_id='workflow_test_receive_event_by_http')

    def check_app_running():
        if False:
            for i in range(10):
                print('nop')
        status = serve.status().applications[common.HTTP_EVENT_PROVIDER_NAME]
        assert status.status == 'RUNNING'
        return True
    wait_for_condition(check_app_running)
    while True:
        res = send_event()
        if res.status_code == 404:
            sleep(0.5)
        else:
            break
    (key, event_msg) = workflow.get_output(workflow_id='workflow_test_receive_event_by_http')
    assert event_msg == 'event_message'

@pytest.mark.parametrize('workflow_start_regular_shared_serve', [{'num_cpus': 4}], indirect=True)
def test_dynamic_event_by_http(workflow_start_regular_shared_serve):
    if False:
        while True:
            i = 10
    'If a workflow has dynamically generated event arguments, it should\n    return the event as if the event was declared statically.\n    '

    def send_event():
        if False:
            i = 10
            return i + 15
        resp = requests.post('http://127.0.0.1:8000/event/send_event/' + 'workflow_test_dynamic_event_by_http', json={'event_key': 'event_key', 'event_payload': 'event_message_dynamic'})
        return resp

    @ray.remote
    def return_dynamically_generated_event():
        if False:
            for i in range(10):
                print('nop')
        event_task = workflow.wait_for_event(HTTPListener, event_key='event_key')
        return workflow.continuation(event_task)
    workflow.run_async(return_dynamically_generated_event.bind(), workflow_id='workflow_test_dynamic_event_by_http')

    def check_app_running():
        if False:
            while True:
                i = 10
        status = serve.status().applications[common.HTTP_EVENT_PROVIDER_NAME]
        assert status.status == 'RUNNING'
        return True
    wait_for_condition(check_app_running)
    while True:
        res = send_event()
        if res.status_code == 404:
            sleep(0.5)
        else:
            break
    (key, event_msg) = workflow.get_output(workflow_id='workflow_test_dynamic_event_by_http')
    assert event_msg == 'event_message_dynamic'
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))
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
def test_multiple_events_by_http(workflow_start_regular_shared_serve):
    if False:
        for i in range(10):
            print('nop')
    'If a workflow has multiple event arguments, it should wait for them at the\n    same time.\n    '

    def send_event1():
        if False:
            return 10
        resp = requests.post('http://127.0.0.1:8000/event/send_event/' + 'workflow_test_multiple_event_by_http', json={'event_key': 'e1', 'event_payload': 'hello'})
        return resp

    def send_event2():
        if False:
            while True:
                i = 10
        sleep(0.5)
        resp = requests.post('http://127.0.0.1:8000/event/send_event/' + 'workflow_test_multiple_event_by_http', json={'event_key': 'e2', 'event_payload': 'world'})
        return resp

    @ray.remote
    def trivial_task(arg1, arg2):
        if False:
            while True:
                i = 10
        return f'{arg1[1]} {arg2[1]}'
    event1_promise = workflow.wait_for_event(HTTPListener, event_key='e1')
    event2_promise = workflow.wait_for_event(HTTPListener, event_key='e2')
    workflow.run_async(trivial_task.bind(event1_promise, event2_promise), workflow_id='workflow_test_multiple_event_by_http')

    def check_app_running():
        if False:
            i = 10
            return i + 15
        status = serve.status().applications[common.HTTP_EVENT_PROVIDER_NAME]
        assert status.status == 'RUNNING'
        return True
    wait_for_condition(check_app_running)
    while True:
        res = send_event1()
        if res.status_code == 404:
            sleep(0.5)
        else:
            break
    while True:
        res = send_event2()
        if res.status_code == 404:
            sleep(0.5)
        else:
            break
    event_msg = workflow.get_output(workflow_id='workflow_test_multiple_event_by_http')
    assert event_msg == 'hello world'
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))
import os
import sys
import gradio as gr
import pytest
import requests
import ray
from ray import serve
from ray._private.test_utils import wait_for_condition
from ray.serve.gradio_integrations import GradioIngress, GradioServer

@pytest.fixture
def serve_start_shutdown():
    if False:
        for i in range(10):
            print('nop')
    ray.init()
    serve.start()
    yield
    serve.shutdown()
    ray.shutdown()

@pytest.mark.parametrize('use_user_defined_class', [False, True])
def test_gradio_ingress_correctness(serve_start_shutdown, use_user_defined_class: bool):
    if False:
        print('Hello World!')
    '\n    Ensure a Gradio app deployed to a cluster through GradioIngress still\n    produces the correct output.\n    '

    def greet(name):
        if False:
            while True:
                i = 10
        return f'Good morning {name}!'
    if use_user_defined_class:

        @serve.deployment
        class UserDefinedGradioServer(GradioIngress):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(lambda : gr.Interface(fn=greet, inputs='text', outputs='text'))
        app = UserDefinedGradioServer.bind()
    else:
        app = GradioServer.bind(lambda : gr.Interface(fn=greet, inputs='text', outputs='text'))
    serve.run(app)
    test_input = 'Alice'
    response = requests.post('http://127.0.0.1:8000/api/predict/', json={'data': [test_input]})
    assert response.status_code == 200 and response.json()['data'][0] == greet(test_input)

def test_gradio_ingress_scaling(serve_start_shutdown):
    if False:
        print('Hello World!')
    '\n    Check that a Gradio app that has been deployed to a cluster through\n    GradioIngress scales as needed, i.e. separate client requests are served by\n    different replicas.\n    '

    def f(*args):
        if False:
            for i in range(10):
                print('nop')
        return os.getpid()
    app = GradioServer.options(num_replicas=2).bind(lambda : gr.Interface(fn=f, inputs='text', outputs='text'))
    serve.run(app)

    def two_pids_returned():
        if False:
            for i in range(10):
                print('nop')

        @ray.remote
        def get_pid_from_request():
            if False:
                return 10
            r = requests.post('http://127.0.0.1:8000/api/predict/', json={'data': ['input']})
            r.raise_for_status()
            return r.json()['data'][0]
        return len(set(ray.get([get_pid_from_request.remote() for _ in range(10)]))) == 2
    wait_for_condition(two_pids_returned)
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))
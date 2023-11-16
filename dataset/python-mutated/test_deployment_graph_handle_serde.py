import sys
import pytest
import ray
from ray import serve
from ray.serve._private.deployment_graph_build import build as pipeline_build
from ray.serve.dag import InputNode

@serve.deployment
def func():
    if False:
        print('Hello World!')
    pass

@serve.deployment
class Driver:

    def __init__(self, *args):
        if False:
            return 10
        pass

    def __call__(self, *args):
        if False:
            while True:
                i = 10
        pass

def test_environment_start():
    if False:
        print('Hello World!')
    "Make sure that in the beginning ray hasn't been started"
    assert not ray.is_initialized()

def test_func_building():
    if False:
        return 10
    dag = func.bind()
    assert len(pipeline_build(dag)) == 1

def test_class_building():
    if False:
        print('Hello World!')
    dag = Driver.bind()
    assert len(pipeline_build(dag)) == 1

def test_dag_building():
    if False:
        while True:
            i = 10
    dag = Driver.bind(func.bind())
    assert len(pipeline_build(dag)) == 2

def test_nested_building():
    if False:
        for i in range(10):
            print('nop')
    with InputNode() as inp:
        out = func.bind(inp)
        out = Driver.bind().__call__.bind(out)
        out = func.bind(out)
    dag = Driver.bind(out, func.bind())
    assert len(pipeline_build(dag)) == 5

def test_environment_end():
    if False:
        i = 10
        return i + 15
    "Make sure that in the end ray hasn't been started"
    assert not ray.is_initialized()
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))
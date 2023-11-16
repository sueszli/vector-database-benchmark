from typing import List, Dict
from ray.tests.conftest import *
import pytest
import numpy as np
import ray
from ray import ObjectRef
from ray import workflow

def test_objectref_inputs(workflow_start_regular_shared):
    if False:
        print('Hello World!')
    from ray.workflow.tests.utils import skip_client_mode_test
    skip_client_mode_test()

    @ray.remote
    def nested_workflow(n: int):
        if False:
            for i in range(10):
                print('nop')
        if n <= 0:
            return 'nested'
        else:
            return workflow.continuation(nested_workflow.bind(n - 1))

    @ray.remote
    def deref_check(u: int, x: str, y: List[str], z: List[Dict[str, str]]):
        if False:
            for i in range(10):
                print('nop')
        try:
            return (u == 42 and x == 'nested' and isinstance(y[0], ray.ObjectRef) and (ray.get(y) == ['nested']) and isinstance(z[0]['output'], ray.ObjectRef) and (ray.get(z[0]['output']) == 'nested'), f'{u}, {x}, {y}, {z}')
        except Exception as e:
            return (False, str(e))
    (output, s) = workflow.run(deref_check.bind(ray.put(42), nested_workflow.bind(10), [nested_workflow.bind(9)], [{'output': nested_workflow.bind(7)}]))
    assert output is True, s

def test_objectref_outputs(workflow_start_regular_shared):
    if False:
        while True:
            i = 10

    @ray.remote
    def nested_ref():
        if False:
            return 10
        return ray.put(42)

    @ray.remote
    def nested_ref_workflow():
        if False:
            return 10
        return nested_ref.remote()

    @ray.remote
    def return_objectrefs() -> List[ObjectRef]:
        if False:
            print('Hello World!')
        return [ray.put(x) for x in range(5)]
    single = workflow.run(nested_ref_workflow.bind())
    assert ray.get(ray.get(single)) == 42
    multi = workflow.run(return_objectrefs.bind())
    assert ray.get(multi) == list(range(5))

@pytest.mark.skip(reason='There is a bug in Ray DAG that makes it serializable.')
def test_object_deref(workflow_start_regular_shared):
    if False:
        print('Hello World!')

    @ray.remote
    def empty_list():
        if False:
            return 10
        return [1]

    @ray.remote
    def receive_workflow(workflow):
        if False:
            return 10
        pass

    @ray.remote
    def return_workflow():
        if False:
            print('Hello World!')
        return empty_list.bind()

    @ray.remote
    def return_data() -> ray.ObjectRef:
        if False:
            print('Hello World!')
        return ray.put(np.ones(4096))

    @ray.remote
    def receive_data(data: 'ray.ObjectRef[np.ndarray]'):
        if False:
            i = 10
            return i + 15
        return ray.get(data)
    x = empty_list.bind()
    with pytest.raises(ValueError):
        ray.put(x)
    with pytest.raises(ValueError):
        ray.get(receive_workflow.remote(x))
    with pytest.raises(ValueError):
        ray.get(return_workflow.remote())
    obj = return_data.bind()
    arr: np.ndarray = workflow.run(receive_data.bind(obj))
    assert np.array_equal(arr, np.ones(4096))
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))
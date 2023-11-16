import numpy as np
from syft import ActionObject
from syft.client.api import SyftAPICall
from syft.service.action.action_object import Action
from syft.service.response import SyftError
from syft.types.uid import LineageID

def test_actionobject_method(worker):
    if False:
        while True:
            i = 10
    root_domain_client = worker.root_client
    action_store = worker.get_service('actionservice').store
    obj = ActionObject.from_obj('abc')
    pointer = root_domain_client.api.services.action.set(obj)
    assert len(action_store.data) == 1
    res = pointer.capitalize()
    assert len(action_store.data) == 2
    assert res[0] == 'A'

def test_lib_function_action(worker):
    if False:
        return 10
    root_domain_client = worker.root_client
    numpy_client = root_domain_client.api.lib.numpy
    res = numpy_client.zeros_like([1, 2, 3])
    assert isinstance(res, ActionObject)
    assert all(res == np.array([0, 0, 0]))
    assert len(worker.get_service('actionservice').store.data) > 0

def test_call_lib_function_action2(worker):
    if False:
        return 10
    root_domain_client = worker.root_client
    assert root_domain_client.api.lib.numpy.add(1, 2) == 3

def test_lib_class_init_action(worker):
    if False:
        print('Hello World!')
    root_domain_client = worker.root_client
    numpy_client = root_domain_client.api.lib.numpy
    res = numpy_client.float32(4.0)
    assert isinstance(res, ActionObject)
    assert res == np.float32(4.0)
    assert len(worker.get_service('actionservice').store.data) > 0

def test_call_lib_wo_permission(worker):
    if False:
        for i in range(10):
            print('nop')
    root_domain_client = worker.root_client
    fname = ActionObject.from_obj('my_fake_file')
    obj1_pointer = fname.send(root_domain_client)
    action = Action(path='numpy', op='fromfile', args=[LineageID(obj1_pointer.id)], kwargs={}, result_id=LineageID())
    kwargs = {'action': action}
    api_call = SyftAPICall(node_uid=worker.id, path='action.execute', args=[], kwargs=kwargs)
    res = root_domain_client.api.make_call(api_call)
    assert isinstance(res, SyftError)

def test_call_lib_custom_signature(worker):
    if False:
        for i in range(10):
            print('nop')
    root_domain_client = worker.root_client
    assert all(root_domain_client.api.lib.numpy.concatenate(([1, 2, 3], [4, 5, 6])).syft_action_data == np.array([1, 2, 3, 4, 5, 6]))
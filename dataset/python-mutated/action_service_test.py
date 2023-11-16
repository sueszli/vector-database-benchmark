from syft.service.action.action_object import ActionObject
from syft.service.context import AuthedServiceContext

def get_auth_ctx(worker):
    if False:
        for i in range(10):
            print('nop')
    return AuthedServiceContext(node=worker, credentials=worker.signing_key.verify_key)

def test_action_service_sanity(worker):
    if False:
        return 10
    service = worker.get_service('actionservice')
    obj = ActionObject.from_obj('abc')
    pointer = service.set(get_auth_ctx(worker), obj).ok()
    assert len(service.store.data) == 1
    res = pointer.capitalize()
    assert res[0] == 'A'
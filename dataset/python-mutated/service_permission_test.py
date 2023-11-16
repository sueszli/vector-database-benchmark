import pytest
from syft import SyftError
from syft.client.api import SyftAPICall

@pytest.fixture
def guest_mock_user(root_verify_key, user_stash, guest_user):
    if False:
        while True:
            i = 10
    result = user_stash.partition.set(root_verify_key, guest_user)
    assert result.is_ok()
    user = result.ok()
    assert user is not None
    return user

def test_call_service_syftapi_with_permission(worker, guest_mock_user, update_user):
    if False:
        return 10
    user_id = guest_mock_user.id
    res = worker.root_client.api.services.user.update(user_id, update_user)
    assert res

def test_call_service_syftapi_no_permission(guest_domain_client):
    if False:
        i = 10
        return i + 15
    with pytest.raises(AttributeError):
        guest_domain_client.api.services.user.get_all()

def test_directly_call_service_with_permission(worker, guest_mock_user, update_user):
    if False:
        for i in range(10):
            print('nop')
    root_domain_client = worker.root_client
    user_id = guest_mock_user.id
    api_call = SyftAPICall(node_uid=root_domain_client.id, path='user.update', args=[user_id, update_user], kwargs={})
    signed_call = api_call.sign(root_domain_client.api.signing_key)
    signed_result = root_domain_client.api.connection.make_call(signed_call)
    result = signed_result.message.data
    assert result

def test_directly_call_service_no_permission(guest_domain_client):
    if False:
        i = 10
        return i + 15
    api_call = SyftAPICall(node_uid=guest_domain_client.id, path='user.get_all', args=[], kwargs={})
    signed_call = api_call.sign(guest_domain_client.api.signing_key)
    signed_result = guest_domain_client.api.connection.make_call(signed_call)
    result = signed_result.message.data
    assert isinstance(result, SyftError)
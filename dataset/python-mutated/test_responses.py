def test_text_response(sync_client):
    if False:
        i = 10
        return i + 15
    resp = sync_client.cat.tasks()
    assert resp.meta.status == 200
    assert isinstance(resp.body, str)
    assert str(resp.body) == str(resp)

def test_object_response(sync_client):
    if False:
        while True:
            i = 10
    resp = sync_client.search(size=1)
    assert isinstance(resp.body, dict)
    assert set(resp) == set(resp.body)
    assert resp.items()
    assert resp.keys()
    assert str(resp) == str(resp.body)
    assert resp['hits'] == resp.body['hits']
    assert type(resp.copy()) is dict

def test_exists_response(sync_client):
    if False:
        print('Hello World!')
    resp = sync_client.indices.exists(index='no')
    assert resp.body is False
    assert not resp
    if resp:
        assert False, "Didn't evaluate to 'False'"
    assert str(resp) == 'False'
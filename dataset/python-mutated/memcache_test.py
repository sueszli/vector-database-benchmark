import webtest
import memcache

def test_memcache(testbed):
    if False:
        i = 10
        return i + 15
    app = webtest.TestApp(memcache.app)
    response = app.get('/memcache')
    assert response.status_int == 200
    assert 'Global: 1' in response.body
    response = app.get('/memcache/a')
    assert response.status_int == 200
    assert 'Global: 2' in response.body
    assert 'a: 1' in response.body
    response = app.get('/memcache/b')
    assert response.status_int == 200
    assert 'Global: 3' in response.body
    assert 'b: 1' in response.body
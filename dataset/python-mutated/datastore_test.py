import webtest
import datastore

def test_datastore(testbed):
    if False:
        while True:
            i = 10
    app = webtest.TestApp(datastore.app)
    response = app.get('/datastore')
    assert response.status_int == 200
    assert 'Global: 1' in response.body
    response = app.get('/datastore/a')
    assert response.status_int == 200
    assert 'Global: 2' in response.body
    assert 'a: 1' in response.body
    response = app.get('/datastore/b')
    assert response.status_int == 200
    assert 'Global: 3' in response.body
    assert 'b: 1' in response.body
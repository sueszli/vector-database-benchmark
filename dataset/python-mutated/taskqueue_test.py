import webtest
import taskqueue

def test_taskqueue(testbed, run_tasks):
    if False:
        i = 10
        return i + 15
    app = webtest.TestApp(taskqueue.app)
    response = app.get('/taskqueue')
    assert response.status_int == 200
    assert 'Global: 0' in response.body
    run_tasks(app)
    response = app.get('/taskqueue')
    assert response.status_int == 200
    assert 'Global: 1' in response.body
    response = app.get('/taskqueue/a')
    assert response.status_int == 200
    assert 'a: 0' in response.body
    run_tasks(app)
    response = app.get('/taskqueue/a')
    assert response.status_int == 200
    assert 'a: 1' in response.body
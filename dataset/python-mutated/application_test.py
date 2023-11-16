import webtest
import application
import worker

def test_all(testbed, run_tasks):
    if False:
        i = 10
        return i + 15
    test_app = webtest.TestApp(application.app)
    test_worker = webtest.TestApp(worker.app)
    response = test_app.get('/')
    assert '0' in response.body
    test_app.post('/enqueue', {'amount': 5})
    run_tasks(test_worker)
    response = test_app.get('/')
    assert '5' in response.body
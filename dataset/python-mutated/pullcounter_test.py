import os
from google.appengine.ext import testbed as gaetestbed
import mock
import webtest
import main

def test_app(testbed):
    if False:
        i = 10
        return i + 15
    key_name = 'foo'
    testbed.init_taskqueue_stub(root_path=os.path.dirname(__file__))
    app = webtest.TestApp(main.app)
    app.post('/', {'key': key_name})
    tq_stub = testbed.get_stub(gaetestbed.TASKQUEUE_SERVICE_NAME)
    tasks = tq_stub.get_filtered_tasks()
    assert len(tasks) == 1
    assert tasks[0].name == 'task1'
    with mock.patch('main.update_counter') as mock_update:
        mock_update.side_effect = RuntimeError()
        app.get('/_ah/start', status=500)
        assert mock_update.called
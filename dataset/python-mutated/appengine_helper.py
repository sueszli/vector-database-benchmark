"""
Common testing tools for Google App Engine tests.
"""
import os
import sys
import tempfile
import pytest
import six

def setup_sdk_imports():
    if False:
        return 10
    'Sets up appengine SDK third-party imports.'
    if six.PY3:
        return
    sdk_path = os.environ.get('GAE_SDK_PATH')
    if not sdk_path:
        return
    if os.path.exists(os.path.join(sdk_path, 'google_appengine')):
        sdk_path = os.path.join(sdk_path, 'google_appengine')
    if 'google' in sys.modules:
        sys.modules['google'].__path__.append(os.path.join(sdk_path, 'google'))
    sys.path.append(sdk_path)
    import dev_appserver
    sys.path.extend(dev_appserver.EXTRA_PATHS)
    import google.appengine.tools.os_compat
    google.appengine.tools.os_compat

def import_appengine_config():
    if False:
        return 10
    'Imports an application appengine_config.py. This is used to\n    mimic the behavior of the runtime.'
    try:
        import appengine_config
        appengine_config
    except ImportError:
        pass

def setup_testbed():
    if False:
        while True:
            i = 10
    'Sets up the GAE testbed and enables common stubs.'
    from google.appengine.datastore import datastore_stub_util
    from google.appengine.ext import testbed as gaetestbed
    tb = gaetestbed.Testbed()
    tb.activate()
    policy = datastore_stub_util.PseudoRandomHRConsistencyPolicy(probability=1.0)
    tb.init_datastore_v3_stub(datastore_file=tempfile.mkstemp()[1], consistency_policy=policy)
    tb.init_memcache_stub()
    tb.init_urlfetch_stub()
    tb.init_app_identity_stub()
    tb.init_blobstore_stub()
    tb.init_user_stub()
    tb.init_logservice_stub()
    tb.init_taskqueue_stub()
    tb.taskqueue_stub = tb.get_stub(gaetestbed.TASKQUEUE_SERVICE_NAME)
    return tb

def run_taskqueue_tasks(testbed, app):
    if False:
        print('Hello World!')
    'Runs tasks that are queued in the GAE taskqueue.'
    from google.appengine.api import namespace_manager
    tasks = testbed.taskqueue_stub.get_filtered_tasks()
    for task in tasks:
        namespace = task.headers.get('X-AppEngine-Current-Namespace', '')
        previous_namespace = namespace_manager.get_namespace()
        try:
            namespace_manager.set_namespace(namespace)
            app.post(task.url, task.extract_params(), headers=dict([(k, v) for (k, v) in task.headers.iteritems() if k.startswith('X-AppEngine')]))
        finally:
            namespace_manager.set_namespace(previous_namespace)

@pytest.fixture
def testbed():
    if False:
        while True:
            i = 10
    'py.test fixture for the GAE testbed.'
    testbed = setup_testbed()
    yield testbed
    testbed.deactivate()

@pytest.fixture
def login(testbed):
    if False:
        print('Hello World!')
    'py.test fixture for logging in GAE users.'

    def _login(email='user@example.com', id='123', is_admin=False):
        if False:
            while True:
                i = 10
        testbed.setup_env(user_email=email, user_id=id, user_is_admin='1' if is_admin else '0', overwrite=True)
    return _login

@pytest.fixture
def run_tasks(testbed):
    if False:
        while True:
            i = 10
    'py.test fixture for running GAE tasks.'

    def _run_tasks(app):
        if False:
            i = 10
            return i + 15
        run_taskqueue_tasks(testbed, app)
    return _run_tasks

def pytest_configure():
    if False:
        i = 10
        return i + 15
    'conftest.py hook function for setting up SDK imports.'
    setup_sdk_imports()

def pytest_runtest_call(item):
    if False:
        for i in range(10):
            print('nop')
    'conftest.py hook for setting up appengine configuration.'
    import_appengine_config()
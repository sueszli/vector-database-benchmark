import ckan.plugins as plugins
from ckan.config.middleware import make_app
from ckan.cli import load_config
from ckan.common import config
_tests_test_request_context = None
_config = config.copy()

def pytest_addoption(parser):
    if False:
        for i in range(10):
            print('nop')
    'Allow using custom config file during tests.\n\n    Catch the exception raised by pytest if  the ``--ckan-ini`` option was\n    already added by the external pytest-ckan package\n    '
    try:
        parser.addoption(u'--ckan-ini', action=u'store')
    except ValueError as e:
        if str(e) == "option names {'--ckan-ini'} already added":
            pass
        else:
            raise

def pytest_sessionstart(session):
    if False:
        while True:
            i = 10
    'Initialize CKAN environment.\n    '
    conf = load_config(session.config.option.ckan_ini)
    global _tests_test_request_context
    app = make_app(conf)
    try:
        flask_app = app.apps['flask_app']._wsgi_app
    except AttributeError:
        flask_app = app._wsgi_app
    _tests_test_request_context = flask_app.test_request_context()
    global _config
    _config = config.copy()

def pytest_runtestloop(session):
    if False:
        for i in range(10):
            print('nop')
    'When all the tests collected, extra plugin may be enabled because python\n    interpreter visits their files.\n\n    Make sure only configured plugins are active when test loop starts.\n    '
    plugins.load_all()

def pytest_runtest_setup(item):
    if False:
        return 10
    'Automatically apply `ckan_config` fixture if test has `ckan_config`\n    mark.\n\n    `ckan_config` mark itself does nothing(as any mark). All actual\n    config changes performed inside `ckan_config` fixture. So let\'s\n    implicitly use `ckan_config` fixture inside any test that patches\n    config object. This will save us from adding\n    `@mark.usefixtures("ckan_config")` every time.\n\n    '
    config.clear()
    config.update(_config)
    custom_config = [mark.args for mark in item.iter_markers(name=u'ckan_config')]
    if custom_config:
        item.fixturenames.append(u'ckan_config')
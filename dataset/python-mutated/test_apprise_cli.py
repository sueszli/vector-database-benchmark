from __future__ import print_function
import re
from unittest import mock
import requests
import json
from inspect import cleandoc
from os.path import dirname
from os.path import join
from apprise import cli
from apprise import NotifyBase
from apprise.common import NOTIFY_CUSTOM_MODULE_MAP
from apprise.utils import PATHS_PREVIOUSLY_SCANNED
from click.testing import CliRunner
from apprise.common import NOTIFY_SCHEMA_MAP
from apprise.utils import environ
from apprise.plugins import __load_matrix
from apprise.plugins import __reset_matrix
from apprise.AppriseLocale import gettext_lazy as _
from importlib import reload
import logging
logging.disable(logging.CRITICAL)

def test_apprise_cli_nux_env(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    CLI: Nux Environment\n\n    '

    class GoodNotification(NotifyBase):

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(*args, **kwargs)

        def notify(self, **kwargs):
            if False:
                while True:
                    i = 10
            return True

        async def async_notify(self, **kwargs):
            return True

        def url(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return 'good://'

    class BadNotification(NotifyBase):

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            super().__init__(*args, **kwargs)

        async def async_notify(self, **kwargs):
            return False

        def url(self, *args, **kwargs):
            if False:
                print('Hello World!')
            return 'bad://'
    NOTIFY_SCHEMA_MAP['good'] = GoodNotification
    NOTIFY_SCHEMA_MAP['bad'] = BadNotification
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 1
    result = runner.invoke(cli.main, ['-v'])
    assert result.exit_code == 1
    result = runner.invoke(cli.main, ['-vv'])
    assert result.exit_code == 1
    result = runner.invoke(cli.main, ['-vvv'])
    assert result.exit_code == 1
    result = runner.invoke(cli.main, ['-vvvv'])
    assert result.exit_code == 1
    result = runner.invoke(cli.main, ['-V'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', 'good://localhost'])
    assert result.exit_code == 0
    with mock.patch('requests.post') as mock_post:
        mock_post.return_value = requests.Request()
        mock_post.return_value.status_code = requests.codes.ok
        result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body\\nsNewLine', '-e', 'json://localhost'])
        assert result.exit_code == 0
        assert mock_post.call_count == 1
        json.loads(mock_post.call_args_list[0][1]['data']).get('message', '') == 'test body\nsNewLine'
        mock_post.reset_mock()
        result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body\\nsNewLine', 'json://localhost'])
        assert result.exit_code == 0
        assert mock_post.call_count == 1
        json.loads(mock_post.call_args_list[0][1]['data']).get('message', '') == 'test body\\nsNewLine'
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', 'good://localhost', '--disable-async'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', 'good://localhost', '--debug'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', 'good://localhost', '-D'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-t', 'test title', 'good://localhost'], input='test stdin body\n')
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-t', 'test title', 'good://localhost', '--disable-async'], input='test stdin body\n')
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', 'bad://localhost'])
    assert result.exit_code == 1
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', 'bad://localhost', '-Da'])
    assert result.exit_code == 1
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', 'bad://localhost', '--dry-run'])
    assert result.exit_code == 0
    t = tmpdir.mkdir('apprise-obj').join('apprise')
    buf = '\n    # Include ourselves\n    include {}\n\n    taga,tagb=good://localhost\n    tagc=good://nuxref.com\n    '.format(str(t))
    t.write(buf)
    result = runner.invoke(cli.main, ['-b', 'test config', '--config', str(t)])
    assert result.exit_code == 3
    result = runner.invoke(cli.main, ['-b', 'has taga', '--config', str(t), '--tag', 'taga'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', '--config', str(t), '--tag', 'tagc', '-R', 'invalid'])
    assert result.exit_code == 2
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', '--config', str(t), '--tag', 'tagc', '--recursive-depth'])
    assert result.exit_code == 2
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', '--config', str(t), '--tag', 'tagc', '-R', '0'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', '--config', str(t), '--tag', 'tagc', '--recursion-depth', '5'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', 'has taga OR tagc OR tagd', '--config', str(t), '--tag', 'taga', '--tag', 'tagc', '--tag', 'tagd'])
    assert result.exit_code == 0
    t = tmpdir.mkdir('apprise-obj2').join('apprise-test2')
    buf = '\n    good://localhost/1\n    good://localhost/2\n    good://localhost/3\n    good://localhost/4\n    good://localhost/5\n    myTag=good://localhost/6\n    '
    t.write(buf)
    result = runner.invoke(cli.main, ['-b', 'test config', '--config', str(t)])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', '# test config', '--config', str(t), '-n', 'success'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', 'test config', '--config', str(t), '--notification-type', 'invalid'])
    assert result.exit_code == 2
    result = runner.invoke(cli.main, ['-b', 'test config', '--config', str(t), '--notification-type', 'WARNING'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', '# test config', '--config', str(t), '-i', 'markdown'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', 'test config', '--config', str(t), '--input-format', 'invalid'])
    assert result.exit_code == 2
    result = runner.invoke(cli.main, ['-b', '# test config', '--config', str(t), '--input-format', 'HTML'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', 'test config', '--config', str(t), '--dry-run'])
    assert result.exit_code == 0
    lines = re.split('[\\r\\n]', result.output.strip())
    assert len(lines) == 5
    for i in range(0, 5):
        assert lines[i].endswith('good://')
    result = runner.invoke(cli.main, ['-b', 'has mytag', '--config', str(t), '--tag', 'mytag'])
    assert result.exit_code == 3
    result = runner.invoke(cli.main, ['-b', 'has mytag', '--config', str(t), '--tag', 'mytag', '--dry-run'])
    assert result.exit_code == 3
    result = runner.invoke(cli.main, ['-b', 'has myTag', '--config', str(t), '--attach', join(dirname(__file__), 'var', 'apprise-test.gif'), '--tag', 'myTag'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', 'has myTag', '--config', str(t), '--tag', 'myTag', '--dry-run'])
    assert result.exit_code == 0
    t2 = tmpdir.mkdir('apprise-obj-env').join('apprise')
    buf = '\n    # A general one\n    good://localhost\n\n    # A failure (if we use the fail tag)\n    fail=bad://localhost\n\n    # A normal one tied to myTag\n    myTag=good://nuxref.com\n    '
    t2.write(buf)
    with environ(APPRISE_URLS='good://localhost'):
        result = runner.invoke(cli.main, ['-b', 'test environment', '--tag', 'mytag'])
        assert result.exit_code == 0
        result = runner.invoke(cli.main, ['-b', 'test environment'])
        assert result.exit_code == 0
    with mock.patch('apprise.cli.DEFAULT_CONFIG_PATHS', []):
        with environ(APPRISE_URLS='      '):
            result = runner.invoke(cli.main, ['-b', 'test environment'])
            assert result.exit_code == 1
    with environ(APPRISE_URLS='bad://localhost'):
        result = runner.invoke(cli.main, ['-b', 'test environment'])
        assert result.exit_code == 1
        result = runner.invoke(cli.main, ['-t', 'test title', '-b', 'test body', 'good://localhost'])
        assert result.exit_code == 0
        result = runner.invoke(cli.main, ['-b', 'has myTag', '--config', str(t2), '--tag', 'myTag'])
        assert result.exit_code == 0
    with environ(APPRISE_CONFIG=str(t2)):
        result = runner.invoke(cli.main, ['-b', 'has myTag', '--tag', 'myTag'])
        assert result.exit_code == 0
    with mock.patch('apprise.cli.DEFAULT_CONFIG_PATHS', []):
        with environ(APPRISE_CONFIG='      '):
            result = runner.invoke(cli.main, ['-b', 'my message'])
            assert result.exit_code == 1
    with environ(APPRISE_CONFIG='garbage/file/path.yaml'):
        result = runner.invoke(cli.main, ['-b', 'my message'])
        assert result.exit_code == 1
        result = runner.invoke(cli.main, ['-b', 'has myTag', '--config', str(t2), '--tag', 'myTag'])
        assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', 'has myTag', '--config', str(t2), 'good://localhost', '--tag', 'fail'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', 'reads the url entry only', '--config', str(t2), 'good://localhost', '--tag', 'fail'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-b', 'reads the url entry only', '--config', str(t2), 'bad://localhost', '--tag', 'myTag'])
    assert result.exit_code == 1
    result = runner.invoke(cli.main, ['-e', '-t', 'test\ntitle', '-b', 'test\nbody', 'good://localhost'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['--interpret-escapes', '-b', 'test\nbody', 'good://localhost'])
    assert result.exit_code == 0

def test_apprise_cli_details(tmpdir):
    if False:
        i = 10
        return i + 15
    '\n    CLI: --details (-l)\n\n    '
    runner = CliRunner()
    result = runner.invoke(cli.main, ['--details'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-l'])
    assert result.exit_code == 0
    __reset_matrix()

    class TestReq01Notification(NotifyBase):
        """
        This class is used to test various requirement configurations
        """
        requirements = {'packages_required': ['cryptography <= 3.4', 'ultrasync'], 'packages_recommended': 'django'}

        def url(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return ''

        def send(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return True
    NOTIFY_SCHEMA_MAP['req01'] = TestReq01Notification

    class TestReq02Notification(NotifyBase):
        """
        This class is used to test various requirement configurations
        """
        enabled = False
        requirements = {'packages_required': None, 'packages_recommended': ['cryptography <= 3.4']}

        def url(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return ''

        def send(self, **kwargs):
            if False:
                return 10
            return True
    NOTIFY_SCHEMA_MAP['req02'] = TestReq02Notification

    class TestReq03Notification(NotifyBase):
        """
        This class is used to test various requirement configurations
        """
        requirements = {'details': _('some specified requirement details'), 'packages_recommended': 'cryptography <= 3.4'}

        def url(self, **kwargs):
            if False:
                print('Hello World!')
            return ''

        def send(self, **kwargs):
            if False:
                while True:
                    i = 10
            return True
    NOTIFY_SCHEMA_MAP['req03'] = TestReq03Notification

    class TestReq04Notification(NotifyBase):
        """
        This class is used to test a case where our requirements is fixed
        to a None
        """
        requirements = None

        def url(self, **kwargs):
            if False:
                i = 10
                return i + 15
            return ''

        def send(self, **kwargs):
            if False:
                print('Hello World!')
            return True
    NOTIFY_SCHEMA_MAP['req04'] = TestReq04Notification

    class TestReq05Notification(NotifyBase):
        """
        This class is used to test a case where only packages_recommended
        is identified
        """
        requirements = {'packages_recommended': 'cryptography <= 3.4'}

        def url(self, **kwargs):
            if False:
                return 10
            return ''

        def send(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return True
    NOTIFY_SCHEMA_MAP['req05'] = TestReq05Notification

    class TestDisabled01Notification(NotifyBase):
        """
        This class is used to test a pre-disabled state
        """
        enabled = False
        service_name = 'na01'

        def url(self, **kwargs):
            if False:
                print('Hello World!')
            return ''

        def notify(self, **kwargs):
            if False:
                while True:
                    i = 10
            return True
    NOTIFY_SCHEMA_MAP['na01'] = TestDisabled01Notification

    class TestDisabled02Notification(NotifyBase):
        """
        This class is used to test a post-disabled state
        """
        service_name = 'na02'

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            super().__init__(**kwargs)
            self.enabled = False

        def url(self, **kwargs):
            if False:
                print('Hello World!')
            return ''

        def notify(self, **kwargs):
            if False:
                while True:
                    i = 10
            return True
    NOTIFY_SCHEMA_MAP['na02'] = TestDisabled02Notification

    class TesEnabled01Notification(NotifyBase):
        """
        This class is just a simple enabled one
        """
        service_name = 'good'

        def url(self, **kwargs):
            if False:
                i = 10
                return i + 15
            return ''

        def send(self, **kwargs):
            if False:
                while True:
                    i = 10
            return True
    NOTIFY_SCHEMA_MAP['good'] = TesEnabled01Notification
    result = runner.invoke(cli.main, ['--details'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['-l'])
    assert result.exit_code == 0
    __reset_matrix()
    __load_matrix()

@mock.patch('requests.post')
def test_apprise_cli_plugin_loading(mock_post, tmpdir):
    if False:
        return 10
    '\n    CLI: --plugin-path (-P)\n\n    '
    mock_post.return_value = requests.Request()
    mock_post.return_value.status_code = requests.codes.ok
    runner = CliRunner()
    PATHS_PREVIOUSLY_SCANNED.clear()
    NOTIFY_CUSTOM_MODULE_MAP.clear()
    result = runner.invoke(cli.main, ['--plugin-path', join(str(tmpdir), 'invalid_path'), '-b', 'test\nbody', 'json://localhost'])
    assert result.exit_code == 0
    assert len(PATHS_PREVIOUSLY_SCANNED) == 0
    assert len(NOTIFY_CUSTOM_MODULE_MAP) == 0
    result = runner.invoke(cli.main, ['--plugin-path', str(tmpdir.mkdir('empty')), '-b', 'test\nbody', 'json://localhost'])
    assert result.exit_code == 0
    assert len(PATHS_PREVIOUSLY_SCANNED) == 1
    assert join(str(tmpdir), 'empty') in PATHS_PREVIOUSLY_SCANNED
    assert len(NOTIFY_CUSTOM_MODULE_MAP) == 0
    PATHS_PREVIOUSLY_SCANNED.clear()
    NOTIFY_CUSTOM_MODULE_MAP.clear()
    notify_hook_a_base = tmpdir.mkdir('random')
    notify_hook_a = notify_hook_a_base.join('myhook01.py')
    notify_hook_a.write(cleandoc('\n    raise ImportError\n    '))
    result = runner.invoke(cli.main, ['--plugin-path', str(notify_hook_a), '-b', 'test\nbody', 'clihook://'])
    assert result.exit_code == 1
    assert len(PATHS_PREVIOUSLY_SCANNED) == 1
    assert str(notify_hook_a) in PATHS_PREVIOUSLY_SCANNED
    assert len(NOTIFY_CUSTOM_MODULE_MAP) == 0
    notify_hook_aa = notify_hook_a_base.join('myhook02.py')
    notify_hook_aa.write(cleandoc('\n    garbage entry\n    '))
    result = runner.invoke(cli.main, ['--plugin-path', str(notify_hook_aa), '-b', 'test\nbody', 'clihook://'])
    assert result.exit_code == 1
    assert len(PATHS_PREVIOUSLY_SCANNED) == 2
    assert str(notify_hook_a) in PATHS_PREVIOUSLY_SCANNED
    assert str(notify_hook_aa) in PATHS_PREVIOUSLY_SCANNED
    assert len(NOTIFY_CUSTOM_MODULE_MAP) == 0
    PATHS_PREVIOUSLY_SCANNED.clear()
    NOTIFY_CUSTOM_MODULE_MAP.clear()
    notify_hook_b = tmpdir.mkdir('goodmodule').join('__init__.py')
    notify_hook_b.write(cleandoc('\n    from apprise.decorators import notify\n\n    # We want to trigger on anyone who configures a call to clihook://\n    @notify(on="clihook")\n    def mywrapper(body, title, notify_type, *args, **kwargs):\n        # A simple test - print to screen\n        print("{}: {} - {}".format(notify_type, title, body))\n\n        # No return (so a return of None) get\'s translated to True\n    '))
    result = runner.invoke(cli.main, ['--plugin-path', str(tmpdir), '-b', 'test body', 'clihook://'])
    assert result.exit_code == 0
    assert len(PATHS_PREVIOUSLY_SCANNED) == 2
    assert str(tmpdir) in PATHS_PREVIOUSLY_SCANNED
    assert join(str(tmpdir), 'goodmodule', '__init__.py') in PATHS_PREVIOUSLY_SCANNED
    assert len(NOTIFY_CUSTOM_MODULE_MAP) == 1
    assert 'clihook' in NOTIFY_SCHEMA_MAP
    key = [k for k in NOTIFY_CUSTOM_MODULE_MAP.keys()][0]
    assert len(NOTIFY_CUSTOM_MODULE_MAP[key]['notify']) == 1
    assert 'clihook' in NOTIFY_CUSTOM_MODULE_MAP[key]['notify']
    assert NOTIFY_CUSTOM_MODULE_MAP[key]['notify']['clihook']['fn_name'] == 'mywrapper'
    assert NOTIFY_CUSTOM_MODULE_MAP[key]['notify']['clihook']['url'] == 'clihook://'
    assert NOTIFY_CUSTOM_MODULE_MAP[key]['notify']['clihook']['name'] == 'Custom - clihook'
    assert isinstance(NOTIFY_CUSTOM_MODULE_MAP[key]['notify']['clihook']['plugin'](), NotifyBase)
    assert NOTIFY_CUSTOM_MODULE_MAP[key]['notify']['clihook']['plugin'] == NOTIFY_SCHEMA_MAP['clihook']
    PATHS_PREVIOUSLY_SCANNED.clear()
    NOTIFY_CUSTOM_MODULE_MAP.clear()
    del NOTIFY_SCHEMA_MAP['clihook']
    result = runner.invoke(cli.main, ['--plugin-path', str(notify_hook_b), '-b', 'test body', 'clihook://'])
    assert result.exit_code == 0
    assert result.stdout.strip() == 'info:  - test body'
    PATHS_PREVIOUSLY_SCANNED.clear()
    NOTIFY_CUSTOM_MODULE_MAP.clear()
    del NOTIFY_SCHEMA_MAP['clihook']
    result = runner.invoke(cli.main, ['--plugin-path', dirname(str(notify_hook_b)), '-b', 'test body', 'clihook://'])
    assert result.exit_code == 0
    assert result.stdout.strip() == 'info:  - test body'
    result = runner.invoke(cli.main, ['--plugin-path', dirname(str(notify_hook_b)), '--plugin-path', str(notify_hook_b), '--details'])
    assert result.exit_code == 0
    PATHS_PREVIOUSLY_SCANNED.clear()
    NOTIFY_CUSTOM_MODULE_MAP.clear()
    del NOTIFY_SCHEMA_MAP['clihook']
    notify_hook_b = tmpdir.mkdir('complex').join('complex.py')
    notify_hook_b.write(cleandoc('\n    from apprise.decorators import notify\n\n    # We can\'t over-ride an element that already exists\n    # in this case json://\n    @notify(on="json")\n    def mywrapper_01(body, title, notify_type, *args, **kwargs):\n        # Return True (same as None)\n        return True\n\n    @notify(on="willfail", name="always failing...")\n    def mywrapper_02(body, title, notify_type, *args, **kwargs):\n        # Simply fail\n        return False\n\n    @notify(on="clihook1", name="the original clihook entry")\n    def mywrapper_03(body, title, notify_type, *args, **kwargs):\n        # Return True\n        return True\n\n    # This is a duplicate o the entry above, so it can not be\n    # loaded...\n    @notify(on="clihook1", name="a duplicate of the clihook entry")\n    def mywrapper_04(body, title, notify_type, *args, **kwargs):\n        # Return True\n        return True\n\n    # This is where things get realy cool... we can not only\n    # define the schema we want to over-ride, but we can define\n    # some default values to pass into our wrapper function to\n    # act as a base before whatever was actually passed in is\n    # applied ontop.... think of it like templating information\n    @notify(on="clihook2://localhost")\n    def mywrapper_05(body, title, notify_type, *args, **kwargs):\n        # Return True\n        return True\n\n\n    # This can\'t load because of the defined schema/on definition\n    @notify(on="", name="an invalid schema was specified")\n    def mywrapper_06(body, title, notify_type, *args, **kwargs):\n        return True\n    '))
    result = runner.invoke(cli.main, ['--plugin-path', join(str(tmpdir), 'complex'), '-b', 'test body', 'clihook://'])
    assert result.exit_code == 1
    assert len(PATHS_PREVIOUSLY_SCANNED) == 2
    assert join(str(tmpdir), 'complex') in PATHS_PREVIOUSLY_SCANNED
    assert join(str(tmpdir), 'complex', 'complex.py') in PATHS_PREVIOUSLY_SCANNED
    assert len(NOTIFY_CUSTOM_MODULE_MAP) == 1
    assert 'willfail' in NOTIFY_SCHEMA_MAP
    assert 'clihook1' in NOTIFY_SCHEMA_MAP
    assert 'clihook2' in NOTIFY_SCHEMA_MAP
    key = [k for k in NOTIFY_CUSTOM_MODULE_MAP.keys()][0]
    assert len(NOTIFY_CUSTOM_MODULE_MAP[key]['notify']) == 3
    assert 'willfail' in NOTIFY_CUSTOM_MODULE_MAP[key]['notify']
    assert 'clihook1' in NOTIFY_CUSTOM_MODULE_MAP[key]['notify']
    assert 'clihook2' in NOTIFY_CUSTOM_MODULE_MAP[key]['notify']
    assert 'json' not in NOTIFY_CUSTOM_MODULE_MAP[key]['notify']
    result = runner.invoke(cli.main, ['--plugin-path', join(str(tmpdir), 'complex'), '-b', 'test body', 'willfail://'])
    assert result.exit_code == 1
    result = runner.invoke(cli.main, ['--plugin-path', join(str(tmpdir), 'complex'), '-b', 'test body', 'clihook1://', 'clihook2://'])
    assert result.exit_code == 0
    result = runner.invoke(cli.main, ['--plugin-path', join(str(tmpdir), 'complex'), '--details'])
    assert 'willfail' in result.stdout
    assert 'always failing...' in result.stdout
    assert 'clihook1' in result.stdout
    assert 'the original clihook entry' in result.stdout
    assert 'a duplicate of the clihook entry' not in result.stdout
    assert 'clihook2' in result.stdout
    assert 'Custom - clihook2' in result.stdout
    assert result.exit_code == 0

@mock.patch('platform.system')
def test_apprise_cli_windows_env(mock_system):
    if False:
        while True:
            i = 10
    '\n    CLI: Windows Environment\n\n    '
    mock_system.return_value = 'Windows'
    reload(cli)
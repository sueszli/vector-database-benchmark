"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import urllib.error
import pytest
import salt.modules.apache as apache
from salt.utils.odict import OrderedDict
from tests.support.mock import MagicMock, mock_open, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {apache: {}}

def test_version():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if return server version (``apachectl -v``)\n    '
    with patch('salt.modules.apache._detect_os', MagicMock(return_value='apachectl')):
        mock = MagicMock(return_value='Server version: Apache/2.4.7')
        with patch.dict(apache.__salt__, {'cmd.run': mock}):
            assert apache.version() == 'Apache/2.4.7'

def test_fullversion():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if return server version (``apachectl -V``)\n    '
    with patch('salt.modules.apache._detect_os', MagicMock(return_value='apachectl')):
        mock = MagicMock(return_value='Server version: Apache/2.4.7')
        with patch.dict(apache.__salt__, {'cmd.run': mock}):
            assert apache.fullversion() == {'compiled_with': [], 'server_version': 'Apache/2.4.7'}

def test_modules():
    if False:
        i = 10
        return i + 15
    '\n    Test if return list of static and shared modules\n    '
    with patch('salt.modules.apache._detect_os', MagicMock(return_value='apachectl')):
        mock = MagicMock(return_value='unixd_module (static)\n                              access_compat_module (shared)')
        with patch.dict(apache.__salt__, {'cmd.run': mock}):
            assert apache.modules() == {'shared': ['access_compat_module'], 'static': ['unixd_module']}

def test_servermods():
    if False:
        while True:
            i = 10
    '\n    Test if return list of modules compiled into the server\n    '
    with patch('salt.modules.apache._detect_os', MagicMock(return_value='apachectl')):
        mock = MagicMock(return_value='core.c\nmod_so.c')
        with patch.dict(apache.__salt__, {'cmd.run': mock}):
            assert apache.servermods() == ['core.c', 'mod_so.c']

def test_directives():
    if False:
        print('Hello World!')
    '\n    Test if return list of directives\n    '
    with patch('salt.modules.apache._detect_os', MagicMock(return_value='apachectl')):
        mock = MagicMock(return_value='Salt')
        with patch.dict(apache.__salt__, {'cmd.run': mock}):
            assert apache.directives() == {'Salt': ''}

def test_vhosts():
    if False:
        i = 10
        return i + 15
    '\n    Test if it shows the virtualhost settings\n    '
    with patch('salt.modules.apache._detect_os', MagicMock(return_value='apachectl')):
        mock = MagicMock(return_value='')
        with patch.dict(apache.__salt__, {'cmd.run': mock}):
            assert apache.vhosts() == {}

def test_signal():
    if False:
        i = 10
        return i + 15
    '\n    Test if return no signal for httpd\n    '
    with patch('salt.modules.apache._detect_os', MagicMock(return_value='apachectl')):
        mock = MagicMock(return_value='')
        with patch.dict(apache.__salt__, {'cmd.run': mock}):
            assert apache.signal(None) is None

def test_signal_args():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if return httpd signal to start, restart, or stop.\n    '
    with patch('salt.modules.apache._detect_os', MagicMock(return_value='apachectl')):
        ret = 'Command: "apachectl -k start" completed successfully!'
        mock = MagicMock(return_value={'retcode': 1, 'stderr': '', 'stdout': ''})
        with patch.dict(apache.__salt__, {'cmd.run_all': mock}):
            assert apache.signal('start') == ret
        mock = MagicMock(return_value={'retcode': 1, 'stderr': 'Syntax OK', 'stdout': ''})
        with patch.dict(apache.__salt__, {'cmd.run_all': mock}):
            assert apache.signal('start') == 'Syntax OK'
        mock = MagicMock(return_value={'retcode': 0, 'stderr': 'Syntax OK', 'stdout': ''})
        with patch.dict(apache.__salt__, {'cmd.run_all': mock}):
            assert apache.signal('start') == 'Syntax OK'
        mock = MagicMock(return_value={'retcode': 1, 'stderr': '', 'stdout': 'Salt'})
        with patch.dict(apache.__salt__, {'cmd.run_all': mock}):
            assert apache.signal('start') == 'Salt'

def test_useradd():
    if False:
        i = 10
        return i + 15
    '\n    Test if it add HTTP user using the ``htpasswd`` command\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(apache.__salt__, {'webutil.useradd': mock}):
        assert apache.useradd('htpasswd', 'salt', 'badpassword') is True

def test_userdel():
    if False:
        return 10
    '\n    Test if it delete HTTP user using the ``htpasswd`` file\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(apache.__salt__, {'webutil.userdel': mock}):
        assert apache.userdel('htpasswd', 'salt') is True

def test_server_status():
    if False:
        print('Hello World!')
    '\n    Test if return get information from the Apache server-status\n    '
    with patch('salt.modules.apache.server_status', MagicMock(return_value={})):
        mock = MagicMock(return_value='')
        with patch.dict(apache.__salt__, {'config.get': mock}):
            assert apache.server_status() == {}

def test_server_status_error():
    if False:
        i = 10
        return i + 15
    '\n    Test if return get error from the Apache server-status\n    '
    mock = MagicMock(side_effect=urllib.error.URLError('error'))
    with patch('urllib.request.urlopen', mock):
        mock = MagicMock(return_value='')
        with patch.dict(apache.__salt__, {'config.get': mock}):
            assert apache.server_status() == 'error'

def test_config():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it create VirtualHost configuration files\n    '
    with patch('salt.modules.apache._parse_config', MagicMock(return_value='Listen 22')):
        with patch('salt.utils.files.fopen', mock_open()):
            assert apache.config('/ports.conf', [{'Listen': '22'}]) == 'Listen 22'

def test__parse_config_dict():
    if False:
        i = 10
        return i + 15
    "\n    Test parsing function which creates configs from dict like (legacy way):\n        - VirtualHost:\n          this: '*:80'\n          ServerName: website.com\n          ServerAlias:\n            - www\n            - dev\n          Directory:\n              this: /var/www/vhosts/website.com\n              Order: Deny,Allow\n              Allow from:\n                - 127.0.0.1\n                - 192.168.100.0/24\n\n    "
    data_in = OrderedDict([('Directory', OrderedDict([('this', '/var/www/vhosts/website.com'), ('Order', 'Deny,Allow'), ('Allow from', ['127.0.0.1', '192.168.100.0/24'])])), ('this', '*:80'), ('ServerName', 'website.com'), ('ServerAlias', ['www', 'dev'])])
    dataout = '<VirtualHost *:80>\n<Directory /var/www/vhosts/website.com>\nOrder Deny,Allow\nAllow from 127.0.0.1\nAllow from 192.168.100.0/24\n\n</Directory>\n\nServerName website.com\nServerAlias www\nServerAlias dev\n\n</VirtualHost>\n'
    parse = apache._parse_config(data_in, 'VirtualHost')
    assert parse == dataout

def test__parse_config_list():
    if False:
        return 10
    "\n    Test parsing function which creates configs from variable structure (list of dicts or\n    list of dicts of dicts/lists) like:\n        - VirtualHost:\n          - this: '*:80'\n          - ServerName: website.com\n          - ServerAlias:\n            - www\n            - dev\n          - Directory:\n              this: /var/www/vhosts/website.com\n              Order: Deny,Allow\n              Allow from:\n                - 127.0.0.1\n                - 192.168.100.0/24\n          - Directory:\n            - this: /var/www/vhosts/website.com/private\n            - Order: Deny,Allow\n            - Allow from:\n              - 127.0.0.1\n              - 192.168.100.0/24\n            - If:\n                this: some condition\n                do: something\n    "
    data_in = [OrderedDict([('ServerName', 'website.com'), ('ServerAlias', ['www', 'dev']), ('Directory', [OrderedDict([('this', '/var/www/vhosts/website.com/private'), ('Order', 'Deny,Allow'), ('Allow from', ['127.0.0.1', '192.168.100.0/24']), ('If', {'this': 'some condition', 'do': 'something'})])]), ('this', '*:80')])]
    dataout = '<VirtualHost *:80>\nServerName website.com\nServerAlias www\nServerAlias dev\n\n<Directory /var/www/vhosts/website.com/private>\nOrder Deny,Allow\nAllow from 127.0.0.1\nAllow from 192.168.100.0/24\n\n<If some condition>\ndo something\n</If>\n\n</Directory>\n\n</VirtualHost>\n'
    parse = apache._parse_config(data_in, 'VirtualHost')
    assert parse == dataout
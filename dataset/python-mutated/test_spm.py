import os
import types
import pytest
import salt.spm
import salt.utils.files
from tests.support.mock import patch

@pytest.fixture
def formula_definition():
    if False:
        return 10
    return {'name': 'formula1', 'version': '1.2', 'release': '2', 'summary': 'test', 'description': 'testing, nothing to see here'}

@pytest.fixture
def formula_contents(formula_definition):
    if False:
        return 10
    return (('FORMULA', 'name: {name}\nversion: {version}\nrelease: {release}\nsummary: {summary}\ndescription: {description}'.format(**formula_definition)), ('modules/mod1.py', '# mod1.py'), ('modules/mod2.py', '# mod2.py'), ('states/state1.sls', '# state1.sls'), ('states/state2.sls', '# state2.sls'))

@pytest.fixture
def formula(formula_definition, formula_contents):
    if False:
        i = 10
        return i + 15
    return types.SimpleNamespace(definition=formula_definition, contents=formula_contents)

class SPMTestUserInterface(salt.spm.SPMUserInterface):
    """
    Unit test user interface to SPMClient
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._status = []
        self._confirm = []
        self._error = []

    def status(self, msg):
        if False:
            print('Hello World!')
        self._status.append(msg)

    def confirm(self, action):
        if False:
            while True:
                i = 10
        self._confirm.append(action)

    def error(self, msg):
        if False:
            print('Hello World!')
        self._error.append(msg)

@pytest.fixture
def minion_config(tmp_path, minion_opts):
    if False:
        print('Hello World!')
    _minion_config = minion_opts.copy()
    _minion_config.update({'spm_logfile': str(tmp_path / 'log'), 'spm_repos_config': str(tmp_path / 'etc' / 'spm.repos'), 'spm_cache_dir': str(tmp_path / 'cache'), 'spm_build_dir': str(tmp_path / 'build'), 'spm_build_exclude': ['.git'], 'spm_db_provider': 'sqlite3', 'spm_files_provider': 'local', 'spm_db': str(tmp_path / 'packages.db'), 'extension_modules': str(tmp_path / 'modules'), 'file_roots': {'base': [str(tmp_path)]}, 'formula_path': str(tmp_path / 'spm'), 'pillar_path': str(tmp_path / 'pillar'), 'reactor_path': str(tmp_path / 'reactor'), 'assume_yes': True, 'root_dir': str(tmp_path), 'force': False, 'verbose': False, 'cache': 'localfs', 'cachedir': str(tmp_path / 'cache'), 'spm_repo_dups': 'ignore', 'spm_share_dir': str(tmp_path / 'share')})
    return _minion_config

@pytest.fixture
def client(minion_config):
    if False:
        while True:
            i = 10
    with patch('salt.client.Caller', return_value=minion_config):
        with patch('salt.client.get_local_client', return_value=minion_config['conf_file']):
            yield salt.spm.SPMClient(SPMTestUserInterface(), minion_config)

@pytest.fixture
def formulas_dir(formula, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    fdir = tmp_path / formula.definition['name']
    fdir.mkdir()
    for (path, contents) in formula.contents:
        path = fdir / path
        (dirname, _) = os.path.split(str(path))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        path.write_text(contents)
    return str(fdir)

def test_build_install(client, formulas_dir, minion_config, formula):
    if False:
        for i in range(10):
            print('nop')
    client.run(['build', formulas_dir])
    pkgpath = client.ui._status[-1].split()[-1]
    assert os.path.exists(pkgpath)
    client.run(['local', 'install', pkgpath])
    for (path, contents) in formula.contents:
        path = os.path.join(minion_config['file_roots']['base'][0], formula.definition['name'], path)
        assert os.path.exists(path)
        with salt.utils.files.fopen(path, 'r') as rfh:
            assert rfh.read().replace('\r\n', '\n') == contents
    client.run(['info', formula.definition['name']])
    lines = client.ui._status[-1].replace('\r\n', '\n').split('\n')
    for (key, line) in (('name', 'Name: {}'), ('version', 'Version: {}'), ('release', 'Release: {}'), ('summary', 'Summary: {}')):
        assert line.format(formula.definition[key]) in lines
    client.ui._error = []
    client.run(['local', 'install', pkgpath])
    assert len(client.ui._error) > 0
    with patch.dict(minion_config, {'force': True}):
        client.ui._error = []
        client.run(['local', 'install', pkgpath])
        assert len(client.ui._error) == 0

def test_repo_paths(client, formulas_dir):
    if False:
        return 10
    client.run(['create_repo', formulas_dir])
    assert len(client.ui._error) == 0

@pytest.mark.parametrize('fail_args', (['bogus', 'command'], ['create_repo'], ['build'], ['build', '/nonexistent/path'], ['info'], ['info', 'not_installed'], ['files'], ['files', 'not_installed'], ['install'], ['install', 'nonexistent.spm'], ['remove'], ['remove', 'not_installed'], ['local', 'bogus', 'command'], ['local', 'info'], ['local', 'info', '/nonexistent/path/junk.spm'], ['local', 'files'], ['local', 'files', '/nonexistent/path/junk.spm'], ['local', 'install'], ['local', 'install', '/nonexistent/path/junk.spm'], ['local', 'list'], ['local', 'list', '/nonexistent/path/junk.spm']))
def test_failure_paths(client, fail_args):
    if False:
        print('Hello World!')
    client.run(fail_args)
    assert len(client.ui._error) > 0
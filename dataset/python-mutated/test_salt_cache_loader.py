"""
Tests for salt.utils.jinja
"""
import copy
import os
import pytest
from jinja2 import Environment, exceptions
import salt.utils.dateutils
import salt.utils.files
import salt.utils.json
import salt.utils.stringutils
import salt.utils.yaml
from salt.utils.jinja import SaltCacheLoader
from tests.support.mock import Mock, call, patch

@pytest.fixture
def minion_opts(tmp_path, minion_opts):
    if False:
        i = 10
        return i + 15
    minion_opts.update({'file_buffer_size': 1048576, 'cachedir': str(tmp_path), 'file_roots': {'test': [str(tmp_path / 'files' / 'test')]}, 'pillar_roots': {'test': [str(tmp_path / 'files' / 'test')]}, 'extension_modules': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extmods')})
    return minion_opts

@pytest.fixture
def hello_simple(template_dir):
    if False:
        while True:
            i = 10
    contents = 'world\n'
    with pytest.helpers.temp_file('hello_simple', directory=template_dir, contents=contents) as hello_simple_filename:
        yield hello_simple_filename

@pytest.fixture
def hello_include(template_dir):
    if False:
        for i in range(10):
            print('nop')
    contents = "{% include 'hello_import' -%}"
    with pytest.helpers.temp_file('hello_include', directory=template_dir, contents=contents) as hello_include_filename:
        yield hello_include_filename

@pytest.fixture
def relative_dir(template_dir):
    if False:
        i = 10
        return i + 15
    relative_dir = template_dir / 'relative'
    relative_dir.mkdir()
    return relative_dir

@pytest.fixture
def relative_rhello(relative_dir):
    if False:
        for i in range(10):
            print('nop')
    contents = "{% from './rmacro' import rmacro with context -%}\n{{ rmacro('Hey') ~ rmacro(a|default('a'), b|default('b')) }}\n"
    with pytest.helpers.temp_file('rhello', directory=relative_dir, contents=contents) as relative_rhello:
        yield relative_rhello

@pytest.fixture
def relative_rmacro(relative_dir):
    if False:
        i = 10
        return i + 15
    contents = "{% from '../macro' import mymacro with context %}\n{% macro rmacro(greeting, greetee='world') -%}\n{{ mymacro(greeting, greetee) }}\n{%- endmacro %}\n"
    with pytest.helpers.temp_file('rmacro', directory=relative_dir, contents=contents) as relative_rmacro:
        yield relative_rmacro

@pytest.fixture
def relative_rescape(relative_dir):
    if False:
        print('Hello World!')
    contents = "{% import '../../rescape' as xfail -%}\n"
    with pytest.helpers.temp_file('rescape', directory=relative_dir, contents=contents) as relative_rescape:
        yield relative_rescape

@pytest.fixture
def get_loader(mock_file_client, minion_opts):
    if False:
        print('Hello World!')

    def run_command(opts=None, saltenv='base', **kwargs):
        if False:
            return 10
        '\n        Now that we instantiate the client in the __init__, we need to mock it\n        '
        if opts is None:
            opts = minion_opts
        mock_file_client.opts = opts
        loader = SaltCacheLoader(opts, saltenv, _file_client=mock_file_client)
        return loader
    return run_command

def get_test_saltenv(get_loader):
    if False:
        print('Hello World!')
    '\n    Setup a simple jinja test environment\n    '
    loader = get_loader(saltenv='test')
    jinja = Environment(loader=loader)
    return (loader._file_client, jinja)

def test_searchpath(minion_opts, get_loader, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    The searchpath is based on the cachedir option and the saltenv parameter\n    '
    opts = copy.deepcopy(minion_opts)
    opts.update({'cachedir': str(tmp_path)})
    loader = get_loader(opts=minion_opts, saltenv='test')
    assert loader.searchpath == [str(tmp_path / 'files' / 'test')]

def test_mockclient(minion_opts, template_dir, hello_simple, get_loader):
    if False:
        while True:
            i = 10
    '\n    A MockFileClient is used that records all file requests normally sent\n    to the master.\n    '
    loader = get_loader(opts=minion_opts, saltenv='test')
    res = loader.get_source(None, 'hello_simple')
    assert len(res) == 3
    assert str(res[0]) == 'world' + os.linesep
    assert res[1] == str(hello_simple)
    assert res[2](), 'Template up to date?'
    assert loader._file_client.requests
    assert loader._file_client.requests[0]['path'] == 'salt://hello_simple'

def test_import(get_loader, hello_import):
    if False:
        return 10
    '\n    You can import and use macros from other files\n    '
    (fc, jinja) = get_test_saltenv(get_loader)
    result = jinja.get_template('hello_import').render()
    assert result == 'Hey world !a b !'
    assert len(fc.requests) == 2
    assert fc.requests[0]['path'] == 'salt://hello_import'
    assert fc.requests[1]['path'] == 'salt://macro'

def test_relative_import(get_loader, relative_rhello, relative_rmacro, relative_rescape, macro_template):
    if False:
        while True:
            i = 10
    '\n    You can import using relative paths\n    issue-13889\n    '
    (fc, jinja) = get_test_saltenv(get_loader)
    tmpl = jinja.get_template(os.path.join('relative', 'rhello'))
    result = tmpl.render()
    assert result == 'Hey world !a b !'
    assert len(fc.requests) == 3
    assert fc.requests[0]['path'] == 'salt://relative/rhello'
    assert fc.requests[1]['path'] == 'salt://relative/rmacro'
    assert fc.requests[2]['path'] == 'salt://macro'
    template = jinja.get_template('relative/rescape')
    pytest.raises(exceptions.TemplateNotFound, template.render)

def test_include(get_loader, hello_include, hello_import):
    if False:
        return 10
    '\n    You can also include a template that imports and uses macros\n    '
    (fc, jinja) = get_test_saltenv(get_loader)
    result = jinja.get_template('hello_include').render()
    assert result == 'Hey world !a b !'
    assert len(fc.requests) == 3
    assert fc.requests[0]['path'] == 'salt://hello_include'
    assert fc.requests[1]['path'] == 'salt://hello_import'
    assert fc.requests[2]['path'] == 'salt://macro'

def test_include_context(get_loader, hello_include, hello_import):
    if False:
        print('Hello World!')
    '\n    Context variables are passes to the included template by default.\n    '
    (_, jinja) = get_test_saltenv(get_loader)
    result = jinja.get_template('hello_include').render(a='Hi', b='Salt')
    assert result == 'Hey world !Hi Salt !'

def test_cached_file_client(get_loader, minion_opts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Multiple instantiations of SaltCacheLoader use the cached file client\n    '
    with patch('salt.channel.client.ReqChannel.factory', Mock()):
        loader_a = SaltCacheLoader(minion_opts)
        loader_b = SaltCacheLoader(minion_opts)
    assert loader_a._file_client is loader_b._file_client

def test_file_client_kwarg(minion_opts, mock_file_client):
    if False:
        return 10
    '\n    A file client can be passed to SaltCacheLoader overriding the any\n    cached file client\n    '
    mock_file_client.opts = minion_opts
    loader = SaltCacheLoader(minion_opts, _file_client=mock_file_client)
    assert loader._file_client is mock_file_client

def test_cache_loader_passed_file_client(minion_opts, mock_file_client):
    if False:
        return 10
    '\n    The shudown method can be called without raising an exception when the\n    file_client does not have a destroy method\n    '
    file_client = Mock()
    with patch('salt.fileclient.get_file_client', return_value=file_client):
        loader = SaltCacheLoader(minion_opts)
        assert loader._file_client is None
        with loader:
            assert loader._file_client is file_client
        assert loader._file_client is None
        assert file_client.mock_calls == [call.destroy()]
    file_client = Mock()
    file_client.opts = {'file_roots': minion_opts['file_roots']}
    with patch('salt.fileclient.get_file_client', return_value=Mock()):
        loader = SaltCacheLoader(minion_opts, _file_client=file_client)
        assert loader._file_client is file_client
        with loader:
            assert loader._file_client is file_client
        assert loader._file_client is file_client
        assert file_client.mock_calls == []
    file_client = Mock()
    file_client.opts = {'file_roots': ''}
    new_file_client = Mock()
    with patch('salt.fileclient.get_file_client', return_value=new_file_client):
        loader = SaltCacheLoader(minion_opts, _file_client=file_client)
        assert loader._file_client is file_client
        with loader:
            assert loader._file_client is not file_client
            assert loader._file_client is new_file_client
        assert loader._file_client is None
        assert file_client.mock_calls == []
        assert new_file_client.mock_calls == [call.destroy()]
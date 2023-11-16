"""
Tests for salt.utils.jinja
"""
import builtins
import datetime
import logging
import os
import pytest
import salt.loader
import salt.utils.dateutils
import salt.utils.files
import salt.utils.json
import salt.utils.stringutils
import salt.utils.yaml
from salt.exceptions import SaltRenderError
from salt.utils.jinja import SaltCacheLoader
from salt.utils.templates import JINJA, render_jinja_tmpl
from tests.support.mock import MagicMock, patch
log = logging.getLogger(__name__)
try:
    import timelib
    HAS_TIMELIB = True
except ImportError:
    HAS_TIMELIB = False

class MockFileClient:
    """
    Does not download files but records any file request for testing
    """

    def __init__(self, loader=None):
        if False:
            return 10
        if loader:
            loader._file_client = self
        self.requests = []

    def get_file(self, template, dest='', makedirs=False, saltenv='base'):
        if False:
            while True:
                i = 10
        self.requests.append({'path': template, 'dest': dest, 'makedirs': makedirs, 'saltenv': saltenv})

@pytest.fixture
def minion_opts(tmp_path, minion_opts):
    if False:
        for i in range(10):
            print('nop')
    minion_opts.update({'cachedir': str(tmp_path), 'file_buffer_size': 1048576, 'file_client': 'local', 'file_ignore_regex': None, 'file_ignore_glob': None, 'file_roots': {'test': [str(tmp_path / 'files' / 'test')]}, 'pillar_roots': {'test': [str(tmp_path / 'files' / 'test')]}, 'fileserver_backend': ['roots'], 'hash_type': 'md5', 'extension_modules': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extmods')})
    return minion_opts

@pytest.fixture
def local_salt():
    if False:
        while True:
            i = 10
    return {}

@pytest.fixture
def non_ascii(template_dir):
    if False:
        print('Hello World!')
    contents = b'Assun\xc3\xa7\xc3\xa3o' + salt.utils.stringutils.to_bytes(os.linesep)
    non_ascii_file = template_dir / 'non-ascii'
    non_ascii_file.write_bytes(contents)
    return non_ascii_file

def test_fallback(minion_opts, local_salt, template_dir):
    if False:
        i = 10
        return i + 15
    '\n    A Template with a filesystem loader is returned as fallback\n    if the file is not contained in the searchpath\n    '
    with pytest.helpers.temp_file('hello_simple', directory=template_dir, contents='world\n') as hello_simple:
        with salt.utils.files.fopen(str(hello_simple)) as fp_:
            out = render_jinja_tmpl(salt.utils.stringutils.to_unicode(fp_.read()), dict(opts=minion_opts, saltenv='test', salt=local_salt))
        assert out == 'world' + os.linesep

def test_fallback_noloader(minion_opts, local_salt, hello_import):
    if False:
        for i in range(10):
            print('nop')
    '\n    A Template with a filesystem loader is returned as fallback\n    if the file is not contained in the searchpath\n    '
    with salt.utils.files.fopen(str(hello_import)) as fp_:
        out = render_jinja_tmpl(salt.utils.stringutils.to_unicode(fp_.read()), dict(opts=minion_opts, saltenv='test', salt=local_salt))
    assert out == 'Hey world !a b !' + os.linesep

def test_saltenv(minion_opts, local_salt, mock_file_client, hello_import):
    if False:
        print('Hello World!')
    '\n    If the template is within the searchpath it can\n    import, include and extend other templates.\n    The initial template is expected to be already cached\n    get_template does not request it from the master again.\n    '
    fc = MockFileClient()
    opts = {'cachedir': minion_opts['cachedir'], 'file_client': 'remote', 'file_roots': minion_opts['file_roots'], 'pillar_roots': minion_opts['pillar_roots']}
    with patch.object(SaltCacheLoader, 'file_client', MagicMock(return_value=fc)):
        with salt.utils.files.fopen(str(hello_import)) as fp_:
            out = render_jinja_tmpl(salt.utils.stringutils.to_unicode(fp_.read()), dict(opts=opts, a='Hi', b='Salt', saltenv='test', salt=local_salt))
        assert out == 'Hey world !Hi Salt !' + os.linesep
        assert fc.requests[0]['path'] == 'salt://macro'

def test_macro_additional_log_for_generalexc(minion_opts, local_salt, hello_import, mock_file_client, template_dir):
    if False:
        while True:
            i = 10
    '\n    If we failed in a macro because of e.g. a TypeError, get\n    more output from trace.\n    '
    expected = 'Jinja error:.*division.*\n.*macrogeneral\\(2\\):\n---\n\\{% macro mymacro\\(\\) -%\\}\n\\{\\{ 1/0 \\}\\}    <======================\n\\{%- endmacro %\\}\n---.*'
    contents = "{% from 'macrogeneral' import mymacro -%}\n{{ mymacro() }}\n"
    macrogeneral_contents = '{% macro mymacro() -%}\n{{ 1/0 }}\n{%- endmacro %}\n'
    with pytest.helpers.temp_file('hello_import_generalerror', directory=template_dir, contents=contents) as hello_import_generalerror:
        with pytest.helpers.temp_file('macrogeneral', directory=template_dir, contents=macrogeneral_contents) as macrogeneral:
            with patch.object(SaltCacheLoader, 'file_client', MagicMock(return_value=mock_file_client)):
                with salt.utils.files.fopen(str(hello_import_generalerror)) as fp_:
                    with pytest.raises(SaltRenderError, match=expected):
                        render_jinja_tmpl(salt.utils.stringutils.to_unicode(fp_.read()), dict(opts=minion_opts, saltenv='test', salt=local_salt))

def test_macro_additional_log_for_undefined(minion_opts, local_salt, mock_file_client, template_dir):
    if False:
        print('Hello World!')
    '\n    If we failed in a macro because of undefined variables, get\n    more output from trace.\n    '
    expected = "Jinja variable 'b' is undefined\n.*macroundefined\\(2\\):\n---\n\\{% macro mymacro\\(\\) -%\\}\n\\{\\{b.greetee\\}\\} <-- error is here    <======================\n\\{%- endmacro %\\}\n---"
    contents = "{% from 'macroundefined' import mymacro -%}\n{{ mymacro() }}\n"
    macroundefined_contents = '{% macro mymacro() -%}\n{{b.greetee}} <-- error is here\n{%- endmacro %}\n'
    with pytest.helpers.temp_file('hello_import_undefined', directory=template_dir, contents=contents) as hello_import_undefined:
        with pytest.helpers.temp_file('macroundefined', directory=template_dir, contents=macroundefined_contents) as macroundefined:
            with patch.object(SaltCacheLoader, 'file_client', MagicMock(return_value=mock_file_client)):
                with salt.utils.files.fopen(str(hello_import_undefined)) as fp_:
                    with pytest.raises(SaltRenderError, match=expected):
                        render_jinja_tmpl(salt.utils.stringutils.to_unicode(fp_.read()), dict(opts=minion_opts, saltenv='test', salt=local_salt))

def test_macro_additional_log_syntaxerror(minion_opts, local_salt, mock_file_client, template_dir):
    if False:
        while True:
            i = 10
    '\n    If  we failed in a macro, get more output from trace.\n    '
    expected = "Jinja syntax error: expected token .*end.*got '-'.*\n.*macroerror\\(2\\):\n---\n# macro\n\\{% macro mymacro\\(greeting, greetee='world'\\) -\\} <-- error is here    <======================\n\\{\\{ greeting ~ ' ' ~ greetee \\}\\} !\n\\{%- endmacro %\\}\n---.*"
    macroerror_contents = "# macro\n{% macro mymacro(greeting, greetee='world') -} <-- error is here\n{{ greeting ~ ' ' ~ greetee }} !\n{%- endmacro %}\n"
    contents = "{% from 'macroerror' import mymacro -%}\n{{ mymacro('Hey') ~ mymacro(a|default('a'), b|default('b')) }}\n"
    with pytest.helpers.temp_file('hello_import_error', directory=template_dir, contents=contents) as hello_import_error:
        with pytest.helpers.temp_file('macroerror', directory=template_dir, contents=macroerror_contents) as macroerror:
            with patch.object(SaltCacheLoader, 'file_client', MagicMock(return_value=mock_file_client)):
                with salt.utils.files.fopen(str(hello_import_error)) as fp_:
                    with pytest.raises(SaltRenderError, match=expected):
                        render_jinja_tmpl(salt.utils.stringutils.to_unicode(fp_.read()), dict(opts=minion_opts, saltenv='test', salt=local_salt))

def test_non_ascii_encoding(minion_opts, local_salt, mock_file_client, non_ascii, hello_import):
    if False:
        return 10
    with patch.object(SaltCacheLoader, 'file_client', MagicMock(return_value=mock_file_client)):
        with salt.utils.files.fopen(str(hello_import)) as fp_:
            out = render_jinja_tmpl(salt.utils.stringutils.to_unicode(fp_.read()), dict(opts={'cachedir': minion_opts['cachedir'], 'file_client': 'remote', 'file_roots': minion_opts['file_roots'], 'pillar_roots': minion_opts['pillar_roots']}, a='Hi', b='Sàlt', saltenv='test', salt=local_salt))
        assert out == salt.utils.stringutils.to_unicode('Hey world !Hi Sàlt !' + os.linesep)
        assert mock_file_client.requests[0]['path'] == 'salt://macro'
        with salt.utils.files.fopen(str(non_ascii), 'rb') as fp_:
            out = render_jinja_tmpl(salt.utils.stringutils.to_unicode(fp_.read(), 'utf-8'), dict(opts={'cachedir': minion_opts['cachedir'], 'file_client': 'remote', 'file_roots': minion_opts['file_roots'], 'pillar_roots': minion_opts['pillar_roots']}, a='Hi', b='Sàlt', saltenv='test', salt=local_salt))
        assert 'Assunção' + os.linesep == out
        assert mock_file_client.requests[0]['path'] == 'salt://macro'

@pytest.mark.skipif(HAS_TIMELIB is False, reason='The `timelib` library is not installed.')
@pytest.mark.parametrize('data_object', [datetime.datetime(2002, 12, 25, 12, 0, 0, 0), '2002/12/25', 1040814000, '1040814000'])
def test_strftime(minion_opts, local_salt, data_object):
    if False:
        return 10
    response = render_jinja_tmpl('{{ "2002/12/25"|strftime }}', dict(opts=minion_opts, saltenv='test', salt=local_salt))
    assert response == '2002-12-25'
    response = render_jinja_tmpl('{{ object|strftime }}', dict(object=data_object, opts=minion_opts, saltenv='test', salt=local_salt))
    assert response == '2002-12-25'
    response = render_jinja_tmpl('{{ object|strftime("%b %d, %Y") }}', dict(object=data_object, opts=minion_opts, saltenv='test', salt=local_salt))
    assert response == 'Dec 25, 2002'
    response = render_jinja_tmpl('{{ object|strftime("%y") }}', dict(object=data_object, opts=minion_opts, saltenv='test', salt=local_salt))
    assert response == '02'

def test_non_ascii(minion_opts, local_salt, non_ascii):
    if False:
        return 10
    out = JINJA(str(non_ascii), opts=minion_opts, saltenv='test', salt=local_salt)
    with salt.utils.files.fopen(out['data'], 'rb') as fp:
        result = salt.utils.stringutils.to_unicode(fp.read(), 'utf-8')
        assert salt.utils.stringutils.to_unicode('Assunção' + os.linesep) == result

def test_get_context_has_enough_context(minion_opts, local_salt):
    if False:
        print('Hello World!')
    template = '1\n2\n3\n4\n5\n6\n7\n8\n9\na\nb\nc\nd\ne\nf'
    context = salt.utils.stringutils.get_context(template, 8)
    expected = '---\n[...]\n3\n4\n5\n6\n7\n8\n9\na\nb\nc\nd\n[...]\n---'
    assert expected == context

def test_get_context_at_top_of_file(minion_opts, local_salt):
    if False:
        print('Hello World!')
    template = '1\n2\n3\n4\n5\n6\n7\n8\n9\na\nb\nc\nd\ne\nf'
    context = salt.utils.stringutils.get_context(template, 1)
    expected = '---\n1\n2\n3\n4\n5\n6\n[...]\n---'
    assert expected == context

def test_get_context_at_bottom_of_file(minion_opts, local_salt):
    if False:
        print('Hello World!')
    template = '1\n2\n3\n4\n5\n6\n7\n8\n9\na\nb\nc\nd\ne\nf'
    context = salt.utils.stringutils.get_context(template, 15)
    expected = '---\n[...]\na\nb\nc\nd\ne\nf\n---'
    assert expected == context

def test_get_context_2_context_lines(minion_opts, local_salt):
    if False:
        return 10
    template = '1\n2\n3\n4\n5\n6\n7\n8\n9\na\nb\nc\nd\ne\nf'
    context = salt.utils.stringutils.get_context(template, 8, num_lines=2)
    expected = '---\n[...]\n6\n7\n8\n9\na\n[...]\n---'
    assert expected == context

def test_get_context_with_marker(minion_opts, local_salt):
    if False:
        i = 10
        return i + 15
    template = '1\n2\n3\n4\n5\n6\n7\n8\n9\na\nb\nc\nd\ne\nf'
    context = salt.utils.stringutils.get_context(template, 8, num_lines=2, marker=' <---')
    expected = '---\n[...]\n6\n7\n8 <---\n9\na\n[...]\n---'
    assert expected == context

def test_render_with_syntax_error(minion_opts, local_salt):
    if False:
        return 10
    template = 'hello\n\n{{ bad\n\nfoo'
    expected = '.*---\\nhello\\n\\n{{ bad\\n\\nfoo    <======================\\n---'
    with pytest.raises(SaltRenderError, match=expected):
        render_jinja_tmpl(template, dict(opts=minion_opts, saltenv='test', salt=local_salt))

def test_render_with_utf8_syntax_error(minion_opts, local_salt):
    if False:
        for i in range(10):
            print('nop')
    with patch.object(builtins, '__salt_system_encoding__', 'utf-8'):
        template = 'hello\n\n{{ bad\n\nfoo한'
        expected = salt.utils.stringutils.to_str('.*---\\nhello\\n\\n{{ bad\\n\\nfoo한    <======================\\n---')
        with pytest.raises(SaltRenderError, match=expected):
            render_jinja_tmpl(template, dict(opts=minion_opts, saltenv='test', salt=local_salt))

def test_render_with_undefined_variable(minion_opts, local_salt):
    if False:
        for i in range(10):
            print('nop')
    template = 'hello\n\n{{ foo }}\n\nfoo'
    expected = "Jinja variable \\'foo\\' is undefined"
    with pytest.raises(SaltRenderError, match=expected):
        render_jinja_tmpl(template, dict(opts=minion_opts, saltenv='test', salt=local_salt))

def test_render_with_undefined_variable_utf8(minion_opts, local_salt):
    if False:
        for i in range(10):
            print('nop')
    template = 'helloí\x95\x9c\n\n{{ foo }}\n\nfoo'
    expected = "Jinja variable \\'foo\\' is undefined"
    with pytest.raises(SaltRenderError, match=expected):
        render_jinja_tmpl(template, dict(opts=minion_opts, saltenv='test', salt=local_salt))

def test_render_with_undefined_variable_unicode(minion_opts, local_salt):
    if False:
        while True:
            i = 10
    template = 'hello한\n\n{{ foo }}\n\nfoo'
    expected = "Jinja variable \\'foo\\' is undefined"
    with pytest.raises(SaltRenderError, match=expected):
        render_jinja_tmpl(template, dict(opts=minion_opts, saltenv='test', salt=local_salt))

def test_relative_include(minion_opts, local_salt, template_dir, hello_import):
    if False:
        for i in range(10):
            print('nop')
    template = "{% include './hello_import' %}"
    expected = 'Hey world !a b !'
    with salt.utils.files.fopen(str(hello_import)) as fp_:
        out = render_jinja_tmpl(template, dict(opts=minion_opts, saltenv='test', salt=local_salt, tpldir=str(template_dir)))
    assert out == expected
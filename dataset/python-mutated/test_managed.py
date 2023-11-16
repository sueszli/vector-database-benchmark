import functools
import hashlib
import os
import shutil
import stat
import types
import pytest
import salt.utils.files
import salt.utils.platform
pytestmark = [pytest.mark.windows_whitelisted]
IS_WINDOWS = salt.utils.platform.is_windows()
BINARY_FILE = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x05\x04\x04\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'

@pytest.fixture
def remote_grail_scene33(webserver, grail_scene33_file, grail_scene33_file_hash):
    if False:
        i = 10
        return i + 15
    return types.SimpleNamespace(file=grail_scene33_file, hash=grail_scene33_file_hash, url=webserver.url('grail/scene33'))

def _format_ids(key, value):
    if False:
        return 10
    return '{}={}'.format(key, value)

def test_managed(file, tmp_path, grail_scene33_file):
    if False:
        for i in range(10):
            print('nop')
    '\n    file.managed\n    '
    name = tmp_path / 'grail_scene33'
    ret = file.managed(name=str(name), source='salt://grail/scene33')
    fileserver_data = grail_scene33_file.read_text()
    local_data = name.read_text()
    assert local_data == fileserver_data
    assert ret.result is True

def test_managed_test(file, tmp_path, grail_scene33_file):
    if False:
        return 10
    '\n    file.managed test interface\n    '
    name = tmp_path / 'grail_scene33'
    ret = file.managed(name=str(name), source='salt://grail/scene33', test=True)
    assert ret.result is None
    assert name.exists() is False

def test_managed_file_mode(file, tmp_path, grail_scene33_file):
    if False:
        print('Hello World!')
    '\n    file.managed, correct file permissions\n    '
    desired_mode = '0o770'
    name = tmp_path / 'grail_scene33'
    ret = file.managed(name=str(name), mode='0770', source='salt://grail/scene33')
    if IS_WINDOWS:
        assert ret.result is False
        assert ret.comment == "The 'mode' option is not supported on Windows"
    else:
        assert ret.result is True
        resulting_mode = stat.S_IMODE(name.stat().st_mode)
        assert oct(resulting_mode) == desired_mode

@pytest.mark.parametrize('mode', [424, 421], ids=functools.partial(_format_ids, 'mode'))
@pytest.mark.parametrize('local', [False, True], ids=functools.partial(_format_ids, 'local'))
@pytest.mark.skip_on_windows(reason='Windows does not report any file modes. Skipping.')
def test_managed_file_mode_keep(file, tmp_path, grail_scene33_file, local, mode):
    if False:
        i = 10
        return i + 15
    '\n    Test using "mode: keep" in a file.managed state\n    '
    name = tmp_path / 'grail_scene33'
    grail_scene33_file.chmod(mode)
    if local is True:
        source = str(grail_scene33_file)
    else:
        source = 'salt://grail/scene33'
    ret = file.managed(name=str(name), mode='keep', source=source, local=local)
    assert ret.result is True
    assert stat.S_IMODE(name.stat().st_mode) == mode

@pytest.mark.parametrize('mode', [424, 421], ids=functools.partial(_format_ids, 'mode'))
@pytest.mark.parametrize('replace', [False, True], ids=functools.partial(_format_ids, 'replace'))
@pytest.mark.skip_on_windows(reason='Windows does not report any file modes. Skipping.')
def test_managed_file_mode_file_exists_replace(file, tmp_path, grail_scene33_file, mode, replace):
    if False:
        for i in range(10):
            print('nop')
    '\n    file.managed, existing file with replace=True, change permissions\n    '
    name = tmp_path / 'grail_scene33'
    grail_scene33_file.chmod(384)
    shutil.copyfile(str(grail_scene33_file), str(name))
    shutil.copymode(str(grail_scene33_file), str(name))
    assert stat.S_IMODE(name.stat().st_mode) != mode
    ret = file.managed(name=str(name), mode=oct(mode), replace=replace, source='salt://grail/scene33')
    assert ret.result is True
    assert stat.S_IMODE(name.stat().st_mode) == mode

def test_managed_file_with_grains_data(file, tmp_path, state_tree, minion_id):
    if False:
        return 10
    '\n    Test to ensure we can render grains data into a managed\n    file.\n    '
    name = tmp_path / 'grains-get-contents.txt'
    tmpl_contents = "\n    {{ salt['grains.get']('id') }}\n    "
    with pytest.helpers.temp_file('grainsget.tmpl', tmpl_contents, state_tree):
        ret = file.managed(name=str(name), source='salt://grainsget.tmpl', template='jinja')
    assert ret.result is True
    assert name.is_file()
    assert name.read_text().strip() == minion_id

@pytest.mark.skip_on_windows(reason='Windows does not report any file modes. Skipping.')
def test_managed_dir_mode(file, tmp_path, grail_scene33_file):
    if False:
        while True:
            i = 10
    '\n    Tests to ensure that file.managed creates directories with the\n    permissions requested with the dir_mode argument\n    '
    desired_mode = 511
    name = tmp_path / 'a' / 'managed_dir_mode_test_file'
    ret = file.managed(name=str(name), source='salt://grail/scene33', mode='600', makedirs=True, dir_mode=oct(desired_mode))
    assert ret.result is True
    assert name.exists()
    assert name.read_text() == grail_scene33_file.read_text()
    resulting_mode = stat.S_IMODE(name.parent.stat().st_mode)
    assert resulting_mode == desired_mode

@pytest.mark.parametrize('show_changes', [False, True], ids=functools.partial(_format_ids, 'show_changes'))
def test_managed_show_changes_false(file, tmp_path, grail_scene33_file, show_changes):
    if False:
        for i in range(10):
            print('nop')
    '\n    file.managed test interface\n    '
    name = tmp_path / 'grail_not_scene33'
    name.write_text('test_managed_show_changes_false\n')
    ret = file.managed(name=str(name), source='salt://grail/scene33', show_changes=False)
    assert ret.result is True
    assert name.exists()
    if show_changes is True:
        assert 'diff' in ret.changes
    else:
        assert ret.changes['diff'] == '<show_changes=False>'

@pytest.mark.skip_on_windows(reason="Don't know how to fix for Windows")
def test_managed_escaped_file_path(file, tmp_path, state_tree):
    if False:
        for i in range(10):
            print('nop')
    "\n    file.managed test that 'salt://|' protects unusual characters in file path\n    "
    funny_file = tmp_path / '?f!le? n@=3&-blah-.file type'
    funny_url = 'salt://|{}'.format(funny_file.name)
    with pytest.helpers.temp_file(funny_file.name, '', state_tree):
        ret = file.managed(name=str(funny_file), source=funny_url)
    assert ret.result is True
    assert funny_file.exists()

@pytest.mark.parametrize('name, contents', [('bool', True), ('str', 'Salt was here.'), ('int', 340282366920938463463374607431768211456), ('float', 1.7518e-45), ('list', [1, 1, 2, 3, 5, 8, 13]), ('dict', {'C': 'charge', 'P': 'parity', 'T': 'time'})])
def test_managed_contents(file, tmp_path, name, contents):
    if False:
        print('Hello World!')
    '\n    test file.managed with contents that is a boolean, string, integer,\n    float, list, and dictionary\n    '
    name = tmp_path / 'managed-{}'.format(name)
    ret = file.managed(name=str(name), contents=contents)
    assert ret.result is True
    assert 'diff' in ret.changes
    assert name.exists()

@pytest.mark.parametrize('contents', ['the contents of the file', 'the contents of the file\n', 'the contents of the file\n\n', 'this is a cookie\nthis is another cookie', 'this is a cookie\nthis is another cookie\n', 'this is a cookie\nthis is another cookie\n\n'])
def test_managed_contents_with_contents_newline(file, tmp_path, contents):
    if False:
        i = 10
        return i + 15
    '\n    test file.managed with contents by using the default contents_newline flag.\n    '
    name = tmp_path / 'foo'
    ret = file.managed(name=str(name), contents=contents, contents_newline=True)
    assert ret.result is True
    assert name.exists()
    expected = contents
    if not expected.endswith('\n'):
        expected += '\n'
    assert name.read_text() == expected

@pytest.mark.skip_on_windows(reason='Windows does not report any file modes. Skipping.')
def test_managed_check_cmd(file, tmp_path):
    if False:
        return 10
    '\n    Test file.managed passing a basic check_cmd kwarg. See Issue #38111.\n    '
    name = tmp_path / 'sudoers'
    ret = file.managed(name=str(name), mode='0440', check_cmd='test -f')
    assert ret.result is True
    assert 'Empty file' in ret.comment
    assert ret.changes == {'new': 'file {} created'.format(name), 'mode': '0440'}

@pytest.mark.parametrize('proto', ['file://', ''])
@pytest.mark.parametrize('dest_file_exists', [False, True])
def test_managed_local_source_with_source_hash(file, tmp_path, grail_scene33_file, grail_scene33_file_hash, proto, dest_file_exists):
    if False:
        return 10
    '\n    Make sure that we enforce the source_hash even with local files\n    '
    name = tmp_path / 'local_source_with_source_hash'
    if dest_file_exists:
        name.touch()
    bad_hash = grail_scene33_file_hash[::-1]
    ret = file.managed(name=str(name), source=proto + str(grail_scene33_file), source_hash='sha256={}'.format(bad_hash))
    assert ret.result is False
    assert not ret.changes
    assert 'does not match actual checksum' in ret.comment
    ret = file.managed(name=str(name), source=proto + str(grail_scene33_file), source_hash='sha256={}'.format(grail_scene33_file_hash))
    assert ret.result is True

@pytest.mark.parametrize('proto', ['file://', ''])
def test_managed_local_source_does_not_exist(file, tmp_path, grail_scene33_file, proto):
    if False:
        while True:
            i = 10
    "\n    Make sure that we exit gracefully when a local source doesn't exist\n    "
    name = tmp_path / 'local_source_does_not_exist'
    ret = file.managed(name=str(name), source=proto + str(grail_scene33_file.with_name('scene99')))
    assert ret.result is False
    assert not ret.changes
    assert 'does not exist' in ret.comment

def test_managed_unicode_jinja_with_tojson_filter(file, tmp_path, state_tree, modules):
    if False:
        for i in range(10):
            print('nop')
    '\n    Using {{ varname }} with a list or dictionary which contains unicode\n    types on Python 2 will result in Jinja rendering the "u" prefix on each\n    string. This tests that using the "tojson" jinja filter will dump them\n    to a format which can be successfully loaded by our YAML loader.\n\n    The two lines that should end up being rendered are meant to test two\n    issues that would trip up PyYAML if the "tojson" filter were not used:\n\n    1. A unicode string type would be loaded as a unicode literal with the\n       leading "u" as well as the quotes, rather than simply being loaded\n       as the proper unicode type which matches the content of the string\n       literal. In other words, u\'foo\' would be loaded literally as\n       u"u\'foo\'". This test includes actual non-ascii unicode in one of the\n       strings to confirm that this also handles these international\n       characters properly.\n\n    2. Any unicode string type (such as a URL) which contains a colon would\n       cause a ScannerError in PyYAML, as it would be assumed to delimit a\n       mapping node.\n\n    Dumping the data structure to JSON using the "tojson" jinja filter\n    should produce an inline data structure which is valid YAML and will be\n    loaded properly by our YAML loader.\n    '
    if salt.utils.platform.is_windows() and os.environ.get('PYTHONUTF8', '0') == '0':
        pytest.skip('Test will fail if PYTHONUTF8=1 is not set on windows')
    test_file = tmp_path / 'test-tojson.txt'
    jinja_template_contents = "\n    {%- for key in ('Die Webseite', 'Der Zucker') -%}\n    {{ key }} ist {{ data[key] }}.\n    {% endfor -%}\n    "
    sls_contents = '\n        {%- set data = \'{"Der Zucker": "süß", "Die Webseite": "https://saltproject.io"}\'|load_json -%}\n        ' + str(test_file) + ':\n          file.managed:\n            - source: salt://template.jinja\n            - template: jinja\n            - context:\n                data: {{ data|tojson }}\n        '
    with pytest.helpers.temp_file('template.jinja', jinja_template_contents, state_tree), pytest.helpers.temp_file('tojson.sls', sls_contents, state_tree):
        ret = modules.state.apply('tojson')
        for state_run in ret:
            assert state_run.result is True
    expected = 'Die Webseite ist https://saltproject.io.\nDer Zucker ist süß.\n\n'
    assert test_file.read_text() == expected

@pytest.mark.parametrize('test', [False, True])
def test_managed_source_hash_indifferent_case(file, tmp_path, state_tree, test):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test passing a source_hash as an uppercase hash.\n\n    This is a regression test for Issue #38914 and Issue #48230 (test=true use).\n    '
    name = tmp_path / 'source_hash_indifferent_case'
    hello_world_contents = 'Hello, World!'
    with pytest.helpers.temp_file('hello_world.txt', hello_world_contents, state_tree) as local_path:
        actual_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()
        shutil.copyfile(str(local_path), str(name))
        ret = file.managed(name=str(name), source=str(local_path), source_hash=actual_hash.upper(), test=test)
        assert ret.result is True
        assert ret.changes == {}

def test_managed_latin1_diff(file, tmp_path, state_tree):
    if False:
        while True:
            i = 10
    '\n    Tests that latin-1 file contents are represented properly in the diff\n    '
    contents = '<html>\n<body>\n{}</body>\n</html>\n'
    testfile = tmp_path / 'issue-48777.html'
    testfile.write_text(contents.format(''))
    with pytest.helpers.temp_file('issue-48777.html', '', state_tree) as src:
        src.write_bytes(contents.format('räksmörgås').encode('latin1'))
        ret = file.managed(name=str(testfile), source='salt://issue-48777.html')
        assert ret.result is True
        assert '+räksmörgås' in ret.changes['diff']

def test_managed_keep_source_false_salt(modules, file, grail_scene33_file, tmp_path):
    if False:
        while True:
            i = 10
    '\n    This test ensures that we properly clean the cached file if keep_source\n    is set to False, for source files using a salt:// URL\n    '
    name = tmp_path / 'grail_scene33'
    source = 'salt://grail/scene33'
    saltenv = 'base'
    ret = modules.cp.is_cached(source, saltenv)
    assert ret == ''
    ret = file.managed(name=str(name), source=source, saltenv=saltenv, keep_source=True)
    assert ret.result is True
    ret = modules.cp.is_cached(source, saltenv)
    assert ret != ''
    name.unlink()
    ret = file.managed(name=str(name), source=source, saltenv=saltenv, keep_source=False)
    assert ret.result is True
    ret = modules.cp.is_cached(source, saltenv)
    assert ret == ''

@pytest.mark.parametrize('requisite', ['onchanges', 'prereq'])
def test_file_managed_requisites(modules, tmp_path, state_tree, requisite):
    if False:
        return 10
    '\n    Test file.managed state with onchanges\n    '
    file1 = tmp_path / 'file1'
    file2 = tmp_path / 'file2'
    sls_contents = '\n    one:\n      file.managed:\n        - name: {file1}\n        - source: salt://testfile\n\n    # This should run because there were changes\n    two:\n      test.succeed_without_changes:\n        - {requisite}:\n          - file: one\n\n    # Run the same state as "one" again, this should not cause changes\n    three:\n      file.managed:\n        - name: {file2}\n        - source: salt://testfile\n\n    # This should not run because there should be no changes\n    four:\n      test.succeed_without_changes:\n        - {requisite}:\n          - file: three\n    '.format(file1=file1, file2=file2, requisite=requisite)
    testfile_contents = 'The test file contents!\n'
    file2.write_text(testfile_contents)
    with pytest.helpers.temp_file('onchanges-prereq.sls', sls_contents, state_tree), pytest.helpers.temp_file('testfile', testfile_contents, state_tree):
        ret = modules.state.apply('onchanges-prereq', test=True)
        assert ret['one'].result is None
        assert ret['three'].result is True
        assert ret['one'].changes
        assert not ret['three'].changes
        assert ret['two'].comment == 'Success!'
        if requisite == 'onchanges':
            expected_comment = 'State was not run because none of the onchanges reqs changed'
        else:
            expected_comment = 'No changes detected'
        assert ret['four'].comment == expected_comment

@pytest.mark.parametrize('prefix', ('', 'file://'))
def test_template_local_file(file, tmp_path, prefix):
    if False:
        i = 10
        return i + 15
    '\n    Test a file.managed state with a local file as the source. Test both\n    with the file:// protocol designation prepended, and without it.\n    '
    source = tmp_path / 'source'
    dest = tmp_path / 'dest'
    source.write_text('{{ foo }}\n')
    ret = file.managed(name=str(dest), source='{}{}'.format(prefix, source), template='jinja', context={'foo': 'Hello world!'})
    assert ret.result is True
    assert dest.read_text() == 'Hello world!\n'

def test_template_local_file_noclobber(file, tmp_path):
    if False:
        i = 10
        return i + 15
    "\n    Test the case where a source file is in the minion's local filesystem,\n    and the source path is the same as the destination path.\n    "
    source = dest = tmp_path / 'source'
    source.write_text('{{ foo }}\n')
    ret = file.managed(name=str(dest), source=str(source), template='jinja', context={'foo': 'Hello world!'})
    assert ret.result is False
    assert 'Source file cannot be the same as destination' in ret.comment

def test_binary_contents(file, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    This tests to ensure that binary contents do not cause a traceback.\n    '
    name = tmp_path / '1px.gif'
    ret = file.managed(name=str(name), contents=BINARY_FILE)
    assert ret.result is True

def test_binary_contents_twice(file, tmp_path):
    if False:
        i = 10
        return i + 15
    '\n    This test ensures that after a binary file is created, salt can confirm\n    that the file is in the correct state.\n    '
    name = tmp_path / '1px.gif'
    ret = file.managed(name=str(name), contents=BINARY_FILE)
    assert ret.result is True
    ret = file.managed(name=str(name), contents=BINARY_FILE)
    assert ret.result is True

def test_issue_8947_utf8_sls(modules, tmp_path, state_tree, subtests):
    if False:
        i = 10
        return i + 15
    '\n    Test some file operation with utf-8 characters on the sls\n\n    This is more generic than just a file test. Feel free to move\n    '
    if salt.utils.platform.is_windows() and os.environ.get('PYTHONUTF8', '0') == '0':
        pytest.skip('Test will fail if PYTHONUTF8=1 is not set on windows')
    korean_1 = '한국어 시험'
    korean_2 = '첫 번째 행'
    korean_3 = '마지막 행'
    test_file = tmp_path / '{}.txt'.format(korean_1)
    with subtests.test('test_file={}'.format(test_file)):
        sls_contents = '\n        some-utf8-file-create:\n          file.managed:\n            - name: {test_file}\n            - contents: {korean_1}\n        '.format(test_file=test_file.as_posix().replace('\\', '/'), korean_1=korean_1)
        with pytest.helpers.temp_file('issue-8947.sls', directory=state_tree, contents=sls_contents):
            ret = modules.state.sls('issue-8947')
            for state_run in ret:
                assert state_run.result is True
        assert test_file.read_text() == '{}\n'.format(korean_1)
    test_file = tmp_path / '{}.txt'.format(korean_2)
    with subtests.test('test_file={}'.format(test_file)):
        sls_contents = '\n        some-utf8-file-create2:\n          file.managed:\n            - name: {test_file}\n            - contents: |\n               {korean_2}\n               {korean_1}\n               {korean_3}\n        '.format(test_file=test_file.as_posix().replace('\\', '/'), korean_1=korean_1, korean_2=korean_2, korean_3=korean_3)
        with pytest.helpers.temp_file('issue-8947.sls', directory=state_tree, contents=sls_contents):
            ret = modules.state.sls('issue-8947')
            for state_run in ret:
                assert state_run.result is True
        assert test_file.read_text() == '{}\n{}\n{}\n'.format(korean_2, korean_1, korean_3)

@pytest.mark.skip_if_not_root
@pytest.mark.skip_on_windows(reason='Windows does not support setuid. Skipping.')
def test_owner_after_setuid(file, modules, tmp_path, state_file_account):
    if False:
        return 10
    '\n    Test to check file user/group after setting setuid or setgid.\n    Because Python os.chown() does reset the setuid/setgid to 0.\n    https://github.com/saltstack/salt/pull/45257\n\n    See also issue #48336\n    '
    desired_file = tmp_path / 'file_with_setuid'
    mode = '4750'
    ret = file.managed(name=str(desired_file), user=state_file_account.username, group=state_file_account.group.name, mode=mode)
    assert ret.result is True
    user_check = modules.file.get_user(str(desired_file))
    assert user_check == state_file_account.username
    group_check = modules.file.get_group(str(desired_file))
    assert group_check == state_file_account.group.name
    mode_check = modules.file.get_mode(str(desired_file))
    assert salt.utils.files.normalize_mode(mode_check) == mode

def test_managed_file_issue_51208(file, tmp_path, state_tree):
    if False:
        return 10
    '\n    Test to ensure we can handle a file with escaped double-quotes\n    '
    vimrc_contents = '\n    set number\n    syntax on\n    set paste\n    set ruler\n    if has("autocmd")\n      au BufReadPost * if line("\'"") > 1 && line("\'"") <= line("$") | exe "normal! g\'"" | endif\n    endif\n\n    '
    with pytest.helpers.temp_file('vimrc.stub', directory=state_tree / 'issue-51208', contents=vimrc_contents) as vimrc_file:
        name = tmp_path / 'issue_51208.txt'
        ret = file.managed(name=str(name), source='salt://issue-51208/vimrc.stub')
        assert ret.result is True
        assert name.read_text() == vimrc_file.read_text()

def test_file_managed_http_source_no_hash(file, tmp_path, remote_grail_scene33):
    if False:
        i = 10
        return i + 15
    '\n    Test a remote file with no hash\n    '
    name = str(tmp_path / 'testfile')
    ret = file.managed(name=name, source=remote_grail_scene33.url, skip_verify=False)
    assert ret.result is False

def test_file_managed_http_source(file, tmp_path, remote_grail_scene33):
    if False:
        while True:
            i = 10
    '\n    Test a remote file with no hash\n    '
    name = str(tmp_path / 'testfile')
    ret = file.managed(name=name, source=remote_grail_scene33.url, source_hash=remote_grail_scene33.hash, skip_verify=False)
    assert ret.result is True

def test_file_managed_http_source_skip_verify(file, tmp_path, remote_grail_scene33):
    if False:
        return 10
    '\n    Test a remote file using skip_verify\n    '
    name = str(tmp_path / 'testfile')
    ret = file.managed(name=name, source=remote_grail_scene33.url, skip_verify=True)
    assert ret.result is True

def test_file_managed_keep_source_false_http(file, tmp_path, remote_grail_scene33, modules):
    if False:
        print('Hello World!')
    '\n    This test ensures that we properly clean the cached file if keep_source\n    is set to False, for source files using an http:// URL\n    '
    name = str(tmp_path / 'testfile')
    ret = file.managed(name=name, source=remote_grail_scene33.url, source_hash=remote_grail_scene33.hash, keep_source=False)
    assert ret.result is True
    ret = modules.cp.is_cached(remote_grail_scene33.url)
    assert not ret, 'File is still cached at {}'.format(ret)

@pytest.mark.parametrize('verify_ssl', [True, False])
def test_verify_ssl_https_source(file, tmp_path, ssl_webserver, verify_ssl):
    if False:
        i = 10
        return i + 15
    '\n    test verify_ssl when its False and True when managing\n    a file with an https source and skip_verify is false.\n    '
    name = tmp_path / 'test_verify_ssl_true.txt'
    source = ssl_webserver.url('this.txt')
    source_hash = f'{source}.sha256'
    ret = file.managed(str(name), source=source, source_hash=source_hash, verify_ssl=verify_ssl, skip_verify=False)
    if verify_ssl is True:
        assert ret.result is False
        assert 'SSL: CERTIFICATE_VERIFY_FAILED' in ret.comment
        assert not name.exists()
    else:
        if IS_WINDOWS and (not os.environ.get('GITHUB_ACTIONS_PIPELINE')):
            pytest.xfail('This test fails when running from Jenkins but not on the GitHub Actions Pipeline')
        assert ret.result is True
        assert ret.changes
        ret.changes.pop('mode', None)
        assert ret.changes == {'diff': 'New file'}
        assert name.exists()

def test_issue_60203(file, tmp_path):
    if False:
        print('Hello World!')
    name = tmp_path / 'test.tar.gz'
    source = 'https://account:dontshowme@notahost.saltstack.io/files/test.tar.gz'
    source_hash = 'https://account:dontshowme@notahost.saltstack.io/files/test.tar.gz.sha256'
    ret = file.managed(str(name), source=source, source_hash=source_hash)
    assert ret.result is False
    assert ret.comment
    assert 'Unable to manage file' in ret.comment
    assert '/files/test.tar.gz.sha256' in ret.comment
    assert 'dontshowme' not in ret.comment
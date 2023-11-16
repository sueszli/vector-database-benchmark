import json
import os
import stat
import time
from ...constants import *
from .. import are_symlinks_supported, are_hardlinks_supported
from ...platformflags import is_win32, is_darwin
from . import cmd, create_regular_file, RK_ENCRYPTION, assert_line_exists, generate_archiver_tests
pytest_generate_tests = lambda metafunc: generate_archiver_tests(metafunc, kinds='local,remote,binary')

def test_basic_functionality(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    create_regular_file(archiver.input_path, 'empty', size=0)
    create_regular_file(archiver.input_path, 'file_unchanged', size=128)
    create_regular_file(archiver.input_path, 'file_removed', size=256)
    create_regular_file(archiver.input_path, 'file_removed2', size=512)
    create_regular_file(archiver.input_path, 'file_replaced', size=1024)
    os.mkdir('input/dir_replaced_with_file')
    os.chmod('input/dir_replaced_with_file', stat.S_IFDIR | 493)
    os.mkdir('input/dir_removed')
    if are_symlinks_supported():
        os.mkdir('input/dir_replaced_with_link')
        os.symlink('input/dir_replaced_with_file', 'input/link_changed')
        os.symlink('input/file_unchanged', 'input/link_removed')
        os.symlink('input/file_removed2', 'input/link_target_removed')
        os.symlink('input/empty', 'input/link_target_contents_changed')
        os.symlink('input/empty', 'input/link_replaced_by_file')
    if are_hardlinks_supported():
        os.link('input/file_replaced', 'input/hardlink_target_replaced')
        os.link('input/empty', 'input/hardlink_contents_changed')
        os.link('input/file_removed', 'input/hardlink_removed')
        os.link('input/file_removed2', 'input/hardlink_target_removed')
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test0', 'input')
    create_regular_file(archiver.input_path, 'file_added', size=2048)
    create_regular_file(archiver.input_path, 'file_empty_added', size=0)
    os.unlink('input/file_replaced')
    create_regular_file(archiver.input_path, 'file_replaced', contents=b'0' * 4096)
    os.unlink('input/file_removed')
    os.unlink('input/file_removed2')
    os.rmdir('input/dir_replaced_with_file')
    create_regular_file(archiver.input_path, 'dir_replaced_with_file', size=8192)
    os.chmod('input/dir_replaced_with_file', stat.S_IFREG | 493)
    os.mkdir('input/dir_added')
    os.rmdir('input/dir_removed')
    if are_symlinks_supported():
        os.rmdir('input/dir_replaced_with_link')
        os.symlink('input/dir_added', 'input/dir_replaced_with_link')
        os.unlink('input/link_changed')
        os.symlink('input/dir_added', 'input/link_changed')
        os.symlink('input/dir_added', 'input/link_added')
        os.unlink('input/link_replaced_by_file')
        create_regular_file(archiver.input_path, 'link_replaced_by_file', size=16384)
        os.unlink('input/link_removed')
    if are_hardlinks_supported():
        os.unlink('input/hardlink_removed')
        os.link('input/file_added', 'input/hardlink_added')
    with open('input/empty', 'ab') as fd:
        fd.write(b'appended_data')
    cmd(archiver, 'create', 'test1a', 'input')
    cmd(archiver, 'create', 'test1b', 'input', '--chunker-params', '16,18,17,4095')

    def do_asserts(output, can_compare_ids, content_only=False):
        if False:
            i = 10
            return i + 15
        lines: list = output.splitlines()
        assert 'file_replaced' in output
        change = 'modified.*B' if can_compare_ids else "modified:  \\(can't get size\\)"
        assert_line_exists(lines, f'{change}.*input/file_replaced')
        assert 'input/file_unchanged' not in output
        if 'BORG_TESTS_IGNORE_MODES' not in os.environ and (not is_win32) and (not content_only):
            assert_line_exists(lines, '[drwxr-xr-x -> -rwxr-xr-x].*input/dir_replaced_with_file')
        assert 'added directory             input/dir_added' in output
        assert 'removed directory           input/dir_removed' in output
        if are_symlinks_supported():
            assert_line_exists(lines, 'changed link.*input/link_changed')
            assert_line_exists(lines, 'added link.*input/link_added')
            assert_line_exists(lines, 'removed link.*input/link_removed')
            if not content_only:
                assert 'input/dir_replaced_with_link' in output
                assert 'input/link_replaced_by_file' in output
            assert 'input/link_target_removed' not in output
        change = 'modified.*0 B' if can_compare_ids else "modified:  \\(can't get size\\)"
        assert_line_exists(lines, f'{change}.*input/empty')
        if are_hardlinks_supported():
            assert_line_exists(lines, f'{change}.*input/hardlink_contents_changed')
        if are_symlinks_supported():
            assert 'input/link_target_contents_changed' not in output
        assert 'added:              2.05 kB input/file_added' in output
        if are_hardlinks_supported():
            assert 'added:              2.05 kB input/hardlink_added' in output
        assert 'added:                  0 B input/file_empty_added' in output
        assert 'removed:              256 B input/file_removed' in output
        if are_hardlinks_supported():
            assert 'removed:              256 B input/hardlink_removed' in output
        if are_hardlinks_supported() and content_only:
            assert 'input/hardlink_target_removed' not in output
            assert 'input/hardlink_target_replaced' not in output

    def do_json_asserts(output, can_compare_ids, content_only=False):
        if False:
            return 10

        def get_changes(filename, data):
            if False:
                i = 10
                return i + 15
            chgsets = [j['changes'] for j in data if j['path'] == filename]
            assert len(chgsets) < 2
            return sum(chgsets, [])
        joutput = [json.loads(line) for line in output.split('\n') if line]
        expected = {'type': 'modified', 'added': 4096, 'removed': 1024} if can_compare_ids else {'type': 'modified'}
        assert expected in get_changes('input/file_replaced', joutput)
        assert not any(get_changes('input/file_unchanged', joutput))
        if 'BORG_TESTS_IGNORE_MODES' not in os.environ and (not is_win32) and (not content_only):
            assert {'type': 'changed mode', 'item1': 'drwxr-xr-x', 'item2': '-rwxr-xr-x'} in get_changes('input/dir_replaced_with_file', joutput)
        assert {'type': 'added directory'} in get_changes('input/dir_added', joutput)
        assert {'type': 'removed directory'} in get_changes('input/dir_removed', joutput)
        if are_symlinks_supported():
            assert {'type': 'changed link'} in get_changes('input/link_changed', joutput)
            assert {'type': 'added link'} in get_changes('input/link_added', joutput)
            assert {'type': 'removed link'} in get_changes('input/link_removed', joutput)
            if not content_only:
                assert any((chg['type'] == 'changed mode' and chg['item1'].startswith('d') and chg['item2'].startswith('l') for chg in get_changes('input/dir_replaced_with_link', joutput))), get_changes('input/dir_replaced_with_link', joutput)
                assert any((chg['type'] == 'changed mode' and chg['item1'].startswith('l') and chg['item2'].startswith('-') for chg in get_changes('input/link_replaced_by_file', joutput))), get_changes('input/link_replaced_by_file', joutput)
            assert not any(get_changes('input/link_target_removed', joutput))
        expected = {'type': 'modified', 'added': 13, 'removed': 0} if can_compare_ids else {'type': 'modified'}
        assert expected in get_changes('input/empty', joutput)
        if are_hardlinks_supported():
            assert expected in get_changes('input/hardlink_contents_changed', joutput)
        if are_symlinks_supported():
            assert not any(get_changes('input/link_target_contents_changed', joutput))
        assert {'added': 2048, 'removed': 0, 'type': 'added'} in get_changes('input/file_added', joutput)
        if are_hardlinks_supported():
            assert {'added': 2048, 'removed': 0, 'type': 'added'} in get_changes('input/hardlink_added', joutput)
        assert {'added': 0, 'removed': 0, 'type': 'added'} in get_changes('input/file_empty_added', joutput)
        assert {'added': 0, 'removed': 256, 'type': 'removed'} in get_changes('input/file_removed', joutput)
        if are_hardlinks_supported():
            assert {'added': 0, 'removed': 256, 'type': 'removed'} in get_changes('input/hardlink_removed', joutput)
        if are_hardlinks_supported() and content_only:
            assert not any(get_changes('input/hardlink_target_removed', joutput))
            assert not any(get_changes('input/hardlink_target_replaced', joutput))
    output = cmd(archiver, 'diff', 'test0', 'test1a')
    do_asserts(output, True)
    output = cmd(archiver, 'diff', 'test0', 'test1b', '--content-only', exit_code=1)
    do_asserts(output, False, content_only=True)
    output = cmd(archiver, 'diff', 'test0', 'test1a', '--json-lines')
    do_json_asserts(output, True)
    output = cmd(archiver, 'diff', 'test0', 'test1a', '--json-lines', '--content-only')
    do_json_asserts(output, True, content_only=True)

def test_time_diffs(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'test_file', size=10)
    cmd(archiver, 'create', 'archive1', 'input')
    time.sleep(0.1)
    os.unlink('input/test_file')
    if is_win32:
        time.sleep(15)
    elif is_darwin:
        time.sleep(1)
    create_regular_file(archiver.input_path, 'test_file', size=15)
    cmd(archiver, 'create', 'archive2', 'input')
    output = cmd(archiver, 'diff', 'archive1', 'archive2', '--format', "'{mtime}{ctime} {path}{NL}'")
    assert 'mtime' in output
    assert 'ctime' in output
    if is_darwin:
        time.sleep(1)
    os.chmod('input/test_file', 511)
    cmd(archiver, 'create', 'archive3', 'input')
    output = cmd(archiver, 'diff', 'archive2', 'archive3', '--format', "'{mtime}{ctime} {path}{NL}'")
    assert 'mtime' not in output
    if not is_win32:
        assert 'ctime' in output
    else:
        assert 'ctime' not in output

def test_sort_option(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_regular_file(archiver.input_path, 'a_file_removed', size=8)
    create_regular_file(archiver.input_path, 'f_file_removed', size=16)
    create_regular_file(archiver.input_path, 'c_file_changed', size=32)
    create_regular_file(archiver.input_path, 'e_file_changed', size=64)
    cmd(archiver, 'create', 'test0', 'input')
    os.unlink('input/a_file_removed')
    os.unlink('input/f_file_removed')
    os.unlink('input/c_file_changed')
    os.unlink('input/e_file_changed')
    create_regular_file(archiver.input_path, 'c_file_changed', size=512)
    create_regular_file(archiver.input_path, 'e_file_changed', size=1024)
    create_regular_file(archiver.input_path, 'b_file_added', size=128)
    create_regular_file(archiver.input_path, 'd_file_added', size=256)
    cmd(archiver, 'create', 'test1', 'input')
    output = cmd(archiver, 'diff', 'test0', 'test1', '--sort', '--content-only')
    expected = ['a_file_removed', 'b_file_added', 'c_file_changed', 'd_file_added', 'e_file_changed', 'f_file_removed']
    assert isinstance(output, str)
    outputs = output.splitlines()
    assert len(outputs) == len(expected)
    assert all((x in line for (x, line) in zip(expected, outputs)))
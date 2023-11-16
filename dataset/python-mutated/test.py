"""
Tests for the filesystem-like abstraction.
"""
import os
from io import UnsupportedOperation
from tempfile import gettempdir, NamedTemporaryFile
from openage.testing.testing import assert_value, assert_raises, result
from .directory import Directory, CaseIgnoringDirectory
from .union import Union
from .wrapper import WriteBlocker, DirectoryCreator

def test_path(root_path, root_dir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test basic functionality of fslike.Path\n    '
    deeper = root_path / "let's go" / 'deeper'
    assert_value(deeper.parent, root_path["let's go"])
    deeper.mkdirs()
    assert_value(deeper.is_dir(), True)
    assert_value(deeper.resolve_native_path().decode(), os.path.join(root_dir, "let's go", 'deeper'))
    insert = deeper['insertion.stuff.test']
    insert.touch()
    assert_value(insert.filesize, 0)
    assert_value(insert.suffix, '.test')
    assert_value(insert.suffixes, ['.stuff', '.test'])
    assert_value(insert.stem, 'insertion.stuff')
    assert_value(insert.with_name('insertion.stuff.test').exists(), True)
    assert_value(insert.with_suffix('.test').exists(), True)
    root_path["let's go"].removerecursive()

def test_union(root_path, root_dir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Union functionality testing.\n\n    Procedure:\n    create and write a file in r\n    create union with w and r mount. r is readonly.\n    read file, should be from r.\n    write file, whould go to w.\n    read file, should be from w.\n    unmount w, file content should be from r again.\n    unmount r, union should be empty now.\n    '
    test_dir_w = os.path.join(root_dir, 'w')
    test_dir_r = os.path.join(root_dir, 'r')
    path_w = DirectoryCreator(Directory(test_dir_w, create_if_missing=True).root).root
    path_r = Directory(test_dir_r, create_if_missing=True).root
    assert_value(path_r['some_file'].is_file(), False)
    with path_r['some_file'].open('wb') as fil:
        fil.write(b'some data')
    with path_r['some_file'].open('rb') as fil:
        assert_value(b'some data', fil.read())
    assert_value(path_r.exists(), True)
    assert_value(path_r.is_dir(), True)
    assert_value(path_r.is_file(), False)
    assert_value(path_r['some_file'].is_file(), True)
    assert_value(path_r.writable(), True)
    path_protected = WriteBlocker(path_r).root
    assert_value(path_protected.writable(), False)
    with assert_raises(UnsupportedOperation):
        result(path_protected.open('wb'))
    target = Union().root
    target.mount(path_protected)
    target.mount(path_w)
    with target['some_file'].open('rb') as fil:
        test_data = fil.read()
    with target['some_file'].open('wb') as fil:
        fil.write(b'we changed it')
    with target['some_file'].open('rb') as fil:
        changed_test_data = fil.read()
    assert_value(test_data != changed_test_data, True)
    assert_value(changed_test_data, b'we changed it')
    assert_value(set(root_path.list()), {b'r', b'w'})
    target.unmount(path_w)
    with (target / 'some_file').open('rb') as fil:
        unchanged_test_data = fil.read()
    assert_value(test_data, unchanged_test_data)
    target.unmount()
    assert_value(target['some_file'].exists(), False)
    assert_value(list(target.list()), [])
    assert_value(len(list(target.iterdir())), 0)

def is_filesystem_case_sensitive():
    if False:
        return 10
    '\n    Utility function to verify if filesystem is case-sensitive.\n    '
    with NamedTemporaryFile() as tmpf:
        return not os.path.exists(tmpf.name.upper())

def test_case_ignoring(root_path, root_dir):
    if False:
        print('Hello World!')
    '\n    Test the case ignoring directory,\n    which mimics the windows filename selection behavior.\n    '
    with root_path['lemme_in'].open('wb') as fil:
        fil.write(b'pwnt')
    ignorecase_dir = CaseIgnoringDirectory(root_dir).root
    with ignorecase_dir['LeMmE_In'].open('rb') as fil:
        assert_value(fil.readable(), True)
        assert_value(fil.writable(), False)
        assert_value(fil.read(), b'pwnt')
    with ignorecase_dir['LeMmE_In'].open('wb') as fil:
        assert_value(fil.readable(), False)
        assert_value(fil.writable(), True)
        fil.write(b'yay')
    with root_path['lemme_in'].open('rb') as fil:
        assert_value(fil.read(), b'yay')
    ignorecase_dir['WeirdCase'].touch()
    assert_value(root_path['weirdcase'].is_file(), True)
    root_path['a'].touch()
    ignorecase_dir['A'].touch()
    if is_filesystem_case_sensitive():
        assert_value(root_path['A'].is_file(), False)
    else:
        assert_value(root_path['A'].is_file(), True)

def test_append(root_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the content append modes.\n    '
    with root_path['appendfile'].open('wb') as fil:
        fil.write(b'just')
    with root_path['appendfile'].open('ab') as fil:
        assert_value(fil.readable(), False)
        assert_value(fil.writable(), True)
        fil.write(b' some')
    with root_path['appendfile'].open('arb') as fil:
        assert_value(fil.readable(), True)
        assert_value(fil.writable(), True)
        fil.write(b' test')
    with root_path['appendfile'].open('rwb') as fil:
        assert_value(fil.readable(), True)
        assert_value(fil.writable(), True)
        assert_value(fil.read(), b'just some test')
        fil.seek(0)
        fil.write(b'overwritten')
        fil.seek(0)
        assert_value(fil.read(), b'overwrittenest')

def test():
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform functionality tests for the filesystem abstraction interface.\n    '
    root_dir = os.path.join(gettempdir(), 'openage_fslike_test')
    root_path = Directory(root_dir, create_if_missing=True).root
    root_path.removerecursive()
    test_path(root_path, root_dir)
    test_union(root_path, root_dir)
    test_case_ignoring(root_path, root_dir)
    test_append(root_path)
    assert_value(root_path.is_dir(), True)
    root_path.removerecursive()
    assert_value(root_path.is_dir(), False)
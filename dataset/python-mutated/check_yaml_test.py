from __future__ import annotations
import pytest
from pre_commit_hooks.check_yaml import main
from testing.util import get_resource_path

@pytest.mark.parametrize(('filename', 'expected_retval'), (('bad_yaml.notyaml', 1), ('ok_yaml.yaml', 0)))
def test_main(filename, expected_retval):
    if False:
        i = 10
        return i + 15
    ret = main([get_resource_path(filename)])
    assert ret == expected_retval

def test_main_allow_multiple_documents(tmpdir):
    if False:
        return 10
    f = tmpdir.join('test.yaml')
    f.write('---\nfoo\n---\nbar\n')
    assert main((str(f),))
    assert not main(('--allow-multiple-documents', str(f)))

def test_fails_even_with_allow_multiple_documents(tmpdir):
    if False:
        return 10
    f = tmpdir.join('test.yaml')
    f.write('[')
    assert main(('--allow-multiple-documents', str(f)))

def test_main_unsafe(tmpdir):
    if False:
        while True:
            i = 10
    f = tmpdir.join('test.yaml')
    f.write('some_foo: !vault |\n    $ANSIBLE_VAULT;1.1;AES256\n    deadbeefdeadbeefdeadbeef\n')
    assert main((str(f),))
    assert not main(('--unsafe', str(f)))

def test_main_unsafe_still_fails_on_syntax_errors(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    f = tmpdir.join('test.yaml')
    f.write('[')
    assert main(('--unsafe', str(f)))
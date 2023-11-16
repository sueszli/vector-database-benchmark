"""
Tests for file.readlink function
"""
import pytest
from salt.exceptions import SaltInvocationError
pytestmark = [pytest.mark.windows_whitelisted]

@pytest.fixture(scope='module')
def file(modules):
    if False:
        for i in range(10):
            print('nop')
    return modules.file

@pytest.fixture(scope='function')
def source():
    if False:
        print('Hello World!')
    with pytest.helpers.temp_file(contents='Source content') as source:
        yield source

def test_readlink(file, source):
    if False:
        i = 10
        return i + 15
    '\n    Test readlink with defaults\n    '
    target = source.parent / 'symlink.lnk'
    target.symlink_to(source)
    try:
        result = file.readlink(path=str(target))
        assert result == str(source)
    finally:
        target.unlink()

def test_readlink_relative_path(file):
    if False:
        print('Hello World!')
    '\n    Test readlink with relative path\n    Should throw a SaltInvocationError\n    '
    with pytest.raises(SaltInvocationError) as exc:
        file.readlink(path='..\\test')
    assert 'Path to link must be absolute' in exc.value.message

def test_readlink_not_a_link(file, source):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test readlink where the path is not a link\n    Should throw a SaltInvocationError\n    '
    with pytest.raises(SaltInvocationError) as exc:
        file.readlink(path=str(source))
    assert 'A valid link was not specified' in exc.value.message

def test_readlink_non_canonical(file, source):
    if False:
        i = 10
        return i + 15
    '\n    Test readlink where there are nested symlinks and canonicalize=False\n    Should resolve to the first symlink\n    '
    intermediate = source.parent / 'intermediate.lnk'
    intermediate.symlink_to(source)
    target = source.parent / 'symlink.lnk'
    target.symlink_to(intermediate)
    try:
        result = file.readlink(path=str(target))
        assert result == str(intermediate)
    finally:
        intermediate.unlink()
        target.unlink()

def test_readlink_canonical(file, source):
    if False:
        print('Hello World!')
    '\n    Test readlink where there are nested symlinks and canonicalize=True\n    Should resolve all nested symlinks returning the path to the source file\n    '
    intermediate = source.parent / 'intermediate.lnk'
    intermediate.symlink_to(source)
    target = source.parent / 'symlink.lnk'
    target.symlink_to(intermediate)
    try:
        result = file.readlink(path=str(target), canonicalize=True)
        assert result == str(source.resolve())
    finally:
        intermediate.unlink()
        target.unlink()
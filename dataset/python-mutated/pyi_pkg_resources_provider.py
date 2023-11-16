import sys
from pkg_resources import resource_exists, resource_isdir, resource_listdir
from pkg_resources import get_provider, DefaultProvider, ZipProvider
pkgname = 'pyi_pkgres_testpkg'
provider = get_provider(pkgname)
is_default = isinstance(provider, DefaultProvider)
is_zip = isinstance(provider, ZipProvider)
is_frozen = getattr(sys, 'frozen', False)
assert is_default or is_zip or is_frozen, 'Unsupported provider type!'
ret = resource_exists(pkgname, '.')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and ret)
assert resource_exists(pkgname, '')
assert resource_exists(pkgname, 'subpkg1')
assert resource_exists(pkgname, 'subpkg2')
assert resource_exists(pkgname, 'subpkg2/subsubpkg21')
assert resource_exists(pkgname, 'subpkg3')
ret = resource_exists(pkgname + '.subpkg1', '.')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and ret)
assert resource_exists(pkgname + '.subpkg1', '')
assert resource_exists(pkgname, 'subpkg1/data')
assert resource_exists(pkgname + '.subpkg1', 'data')
assert resource_exists(pkgname, 'subpkg1/data/extra')
assert resource_exists(pkgname + '.subpkg1', 'data/extra')
assert resource_exists(pkgname, 'subpkg1/data/entry1.txt')
assert resource_exists(pkgname, 'subpkg1/data/extra/extra_entry1.json')
assert not resource_exists(pkgname, 'subpkg1/non-existant')
ret = resource_exists(pkgname, '__init__.py')
assert not is_frozen and ret or (is_frozen and (not ret))
ret = resource_exists(pkgname, '..')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and (not ret))
ret = resource_exists(pkgname + '.subpkg1', '..')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and (not ret))
ret = resource_exists(pkgname + '.a', '.')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and ret)
assert resource_exists(pkgname + '.a', '')
ret = resource_exists(pkgname + '.subpkg1.c', '.')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and ret)
assert resource_exists(pkgname + '.subpkg1.c', '')
ret = resource_isdir(pkgname, '.')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and ret)
assert resource_isdir(pkgname, '')
assert resource_isdir(pkgname, 'subpkg1')
assert resource_isdir(pkgname, 'subpkg2')
assert resource_isdir(pkgname, 'subpkg2/subsubpkg21')
assert resource_isdir(pkgname, 'subpkg3')
ret = resource_isdir(pkgname + '.subpkg1', '.')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and ret)
assert resource_isdir(pkgname + '.subpkg1', '')
assert resource_isdir(pkgname, 'subpkg1/data')
assert resource_isdir(pkgname + '.subpkg1', 'data')
assert resource_isdir(pkgname, 'subpkg1/data/extra')
assert resource_isdir(pkgname + '.subpkg1', 'data/extra')
assert not resource_isdir(pkgname, 'subpkg1/data/entry1.txt')
assert not resource_isdir(pkgname, 'subpkg1/data/extra/extra_entry1.json')
assert not resource_isdir(pkgname, 'subpkg1/non-existant')
assert not resource_isdir(pkgname, '__init__.py')
ret = resource_isdir(pkgname, '..')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and (not ret))
ret = resource_isdir(pkgname + '.subpkg1', '..')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and (not ret))
ret = resource_isdir(pkgname + '.a', '.')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and ret)
assert resource_isdir(pkgname + '.a', '')
ret = resource_isdir(pkgname + '.subpkg1.c', '.')
assert is_default and ret or (is_zip and (not ret)) or (is_frozen and ret)
assert resource_isdir(pkgname + '.subpkg1.c', '')

def _listdir_test(pkgname, path, expected):
    if False:
        for i in range(10):
            print('nop')
    if is_frozen:
        expected = [x for x in expected if not x.endswith('.py')]
    content = resource_listdir(pkgname, path)
    if '__pycache__' in content:
        content.remove('__pycache__')
    assert sorted(content) == sorted(expected)
if is_zip:
    expected = []
else:
    expected = ['__init__.py', 'a.py', 'b.py', 'subpkg1', 'subpkg2', 'subpkg3']
_listdir_test(pkgname, '.', expected)
expected = ['__init__.py', 'a.py', 'b.py', 'subpkg1', 'subpkg2', 'subpkg3']
_listdir_test(pkgname, '', expected)
expected = ['__init__.py', 'c.py', 'd.py', 'data']
_listdir_test(pkgname, 'subpkg1', expected)
expected = ['entry1.txt', 'entry2.md', 'entry3.rst', 'extra']
_listdir_test(pkgname, 'subpkg1/data', expected)
expected = ['entry1.txt', 'entry2.md', 'entry3.rst', 'extra']
_listdir_test(pkgname + '.subpkg1', 'data', expected)
expected = ['extra_entry1.json']
_listdir_test(pkgname + '.subpkg1', 'data/extra', expected)
try:
    content = resource_listdir(pkgname + '.subpkg1', 'data/entry1.txt')
except NotADirectoryError:
    assert is_default
except Exception:
    raise
else:
    assert (is_zip or is_frozen) and content == []
try:
    content = resource_listdir(pkgname, 'non-existant')
except FileNotFoundError:
    assert is_default
except Exception:
    raise
else:
    assert (is_zip or is_frozen) and content == []
try:
    content = resource_listdir(pkgname + '.subpkg1', 'data/non-existant')
except FileNotFoundError:
    assert is_default
except Exception:
    raise
else:
    assert (is_zip or is_frozen) and content == []
content = resource_listdir(pkgname, '..')
assert is_default and pkgname in content or (is_zip and content == []) or (is_frozen and content == [])
if is_default:
    expected = ['__init__.py', 'a.py', 'b.py', 'subpkg1', 'subpkg2', 'subpkg3']
else:
    expected = []
_listdir_test(pkgname + '.subpkg1', '..', expected)
expected = ['__init__.py', 'mod.py', 'subsubpkg21']
_listdir_test(pkgname, 'subpkg2', expected)
expected = ['__init__.py', 'mod.py', 'subsubpkg21']
_listdir_test(pkgname + '.subpkg2', '', expected)
expected = ['__init__.py', 'mod.py']
_listdir_test(pkgname, 'subpkg2/subsubpkg21', expected)
expected = ['__init__.py', 'mod.py']
_listdir_test(pkgname + '.subpkg2', 'subsubpkg21', expected)
expected = ['__init__.py', 'mod.py']
_listdir_test(pkgname + '.subpkg2.subsubpkg21', '', expected)
assert sorted(resource_listdir(pkgname + '.a', '')) == sorted(resource_listdir(pkgname, ''))
assert sorted(resource_listdir(pkgname + '.subpkg1.c', '')) == sorted(resource_listdir(pkgname + '.subpkg1', ''))
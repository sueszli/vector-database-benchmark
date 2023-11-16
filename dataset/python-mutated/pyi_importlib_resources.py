import sys
import pathlib
try:
    import importlib.resources as importlib_resources
    if not hasattr(importlib_resources, 'files'):
        raise ImportError('Built-in importlib.resources is too old!')
    is_builtin = True
    package_name = 'importlib.resources'
    print('Using built-in importlib.resources...')
except ImportError:
    import importlib_resources
    is_builtin = False
    package_name = 'importlib_resources'
    print('Using backported importlib_resources...')
is_frozen = getattr(sys, 'frozen', False)
if hasattr(importlib_resources, 'contents'):
    print(f'Testing {package_name}.contents()...')

    def _contents_test(pkgname, expected):
        if False:
            i = 10
            return i + 15
        if is_frozen:
            expected = [x for x in expected if not x.endswith('.py')]
        content = list(importlib_resources.contents(pkgname))
        if '__pycache__' in content:
            content.remove('__pycache__')
        assert sorted(content) == sorted(expected), f'Content mismatch: {sorted(content)} vs. {sorted(expected)}'
    expected = ['__init__.py', 'a.py', 'b.py', 'subpkg1', 'subpkg2', 'subpkg3']
    if is_frozen:
        expected.remove('subpkg2')
    _contents_test('pyi_pkgres_testpkg', expected)
    expected = ['__init__.py', 'c.py', 'd.py', 'data']
    _contents_test('pyi_pkgres_testpkg.subpkg1', expected)
    if not is_frozen:
        expected = ['__init__.py', 'mod.py', 'subsubpkg21']
        _contents_test('pyi_pkgres_testpkg.subpkg2', expected)
    if not is_frozen:
        expected = ['__init__.py', 'mod.py']
        _contents_test('pyi_pkgres_testpkg.subpkg2.subsubpkg21', expected)
    expected = ['__init__.py', '_datafile.json']
    _contents_test('pyi_pkgres_testpkg.subpkg3', expected)
else:
    print(f'Skipping {package_name}.contents() test...')
if hasattr(importlib_resources, 'is_resource'):
    print(f'Testing {package_name}.is_resource()...')
    assert not importlib_resources.is_resource('pyi_pkgres_testpkg', 'subpkg_nonexistant')
    assert not importlib_resources.is_resource('pyi_pkgres_testpkg.subpkg1', 'nonexistant.txt')
    assert importlib_resources.is_resource('pyi_pkgres_testpkg', '__init__.py') is not is_frozen
    assert importlib_resources.is_resource('pyi_pkgres_testpkg', '__init__.py') is not is_frozen
    assert importlib_resources.is_resource('pyi_pkgres_testpkg', 'a.py') is not is_frozen
    assert importlib_resources.is_resource('pyi_pkgres_testpkg', 'b.py') is not is_frozen
    assert not importlib_resources.is_resource('pyi_pkgres_testpkg', 'subpkg1')
    assert not importlib_resources.is_resource('pyi_pkgres_testpkg', 'subpkg2')
    assert not importlib_resources.is_resource('pyi_pkgres_testpkg', 'subpkg3')
    assert importlib_resources.is_resource('pyi_pkgres_testpkg.subpkg1', '__init__.py') is not is_frozen
    assert not importlib_resources.is_resource('pyi_pkgres_testpkg.subpkg1', 'data')
    try:
        ret = importlib_resources.is_resource('pyi_pkgres_testpkg.subpkg1', 'data/entry1.txt')
    except ValueError:
        pass
    except Exception:
        raise
    else:
        assert False, 'Expected a ValueError!'
    if not is_frozen:
        assert importlib_resources.is_resource('pyi_pkgres_testpkg.subpkg2', 'mod.py')
    assert importlib_resources.is_resource('pyi_pkgres_testpkg.subpkg3', '_datafile.json')
else:
    print(f'Skipping {package_name}.is_resource() test...')
if hasattr(importlib_resources, 'path'):
    print(f'Testing {package_name}.path()...')

    def _path_test(pkgname, resource, expected_data):
        if False:
            i = 10
            return i + 15
        with importlib_resources.path(pkgname, resource) as pth:
            assert isinstance(pth, pathlib.Path)
            with open(pth, 'rb') as fp:
                data = fp.read()
            if expected_data is not None:
                assert data.splitlines() == expected_data.splitlines()
    if not is_frozen:
        expected_data = b'from . import a, b  # noqa: F401\nfrom . import subpkg1, subpkg2, subpkg3  # noqa: F401\n'
        _path_test('pyi_pkgres_testpkg', '__init__.py', expected_data)
    if not is_frozen:
        expected_data = b'#\n'
        _path_test('pyi_pkgres_testpkg.subpkg2', 'mod.py', expected_data)
    expected_data = b'{\n  "_comment": "Data file in supbkg3."\n}\n'
    _path_test('pyi_pkgres_testpkg.subpkg3', '_datafile.json', expected_data)
    try:
        _path_test('pyi_pkgres_testpkg.subpkg1', 'nonexistant.txt', None)
    except FileNotFoundError:
        pass
    except Exception:
        raise
    else:
        assert not is_builtin, 'Expected a FileNotFoundError!'
    try:
        _path_test('pyi_pkgres_testpkg.subpkg1', 'data/entry1.txt', None)
    except ValueError:
        pass
    except Exception:
        raise
    else:
        assert False, 'Expected a ValueError!'
else:
    print(f'Skipping {package_name}.path() test...')
if hasattr(importlib_resources, 'read_binary'):
    print(f'Testing {package_name}.read_binary()...')
    expected_data = b'{\n  "_comment": "Data file in supbkg3."\n}\n'
    data = importlib_resources.read_binary('pyi_pkgres_testpkg.subpkg3', '_datafile.json')
    assert data.splitlines() == expected_data.splitlines()
    if not is_frozen:
        expected_data = b'from . import a, b  # noqa: F401\nfrom . import subpkg1, subpkg2, subpkg3  # noqa: F401\n'
        data = importlib_resources.read_binary('pyi_pkgres_testpkg', '__init__.py')
        assert data.splitlines() == expected_data.splitlines()
    try:
        importlib_resources.read_binary('pyi_pkgres_testpkg.subpkg1', 'nonexistant.txt')
    except FileNotFoundError:
        pass
    except Exception:
        raise
    else:
        assert False, 'Expected a FileNotFoundError!'
    try:
        importlib_resources.read_binary('pyi_pkgres_testpkg.subpkg1', 'data/entry1.txt')
    except ValueError:
        pass
    except Exception:
        raise
    else:
        assert False, 'Expected a ValueError!'
else:
    print(f'Skipping {package_name}.read_binary() test...')
if hasattr(importlib_resources, 'read_text'):
    print(f'Testing {package_name}.read_text()...')
    expected_data = '{\n  "_comment": "Data file in supbkg3."\n}\n'
    data = importlib_resources.read_text('pyi_pkgres_testpkg.subpkg3', '_datafile.json', encoding='utf8')
    assert data.splitlines() == expected_data.splitlines()
    if not is_frozen:
        expected_data = 'from . import a, b  # noqa: F401\nfrom . import subpkg1, subpkg2, subpkg3  # noqa: F401\n'
        data = importlib_resources.read_text('pyi_pkgres_testpkg', '__init__.py', encoding='utf8')
        assert data.splitlines() == expected_data.splitlines()
    try:
        importlib_resources.read_text('pyi_pkgres_testpkg.subpkg1', 'nonexistant.txt', encoding='utf8')
    except FileNotFoundError:
        pass
    except Exception:
        raise
    else:
        assert False, 'Expected a FileNotFoundError!'
    try:
        importlib_resources.read_text('pyi_pkgres_testpkg.subpkg1', 'data/entry1.txt', encoding='utf8')
    except ValueError:
        pass
    except Exception:
        raise
    else:
        assert False, 'Expected a ValueError!'
else:
    print(f'Skipping {package_name}.read_text() test...')
print(f'Testing {package_name}.files() and {package_name}.as_file()...')
pkg_path = importlib_resources.files('pyi_pkgres_testpkg')
subpkg1_path = importlib_resources.files('pyi_pkgres_testpkg.subpkg1')
data_path = subpkg1_path / 'data/entry1.txt'
expected_data = b'Data entry #1 in subpkg1/data.\n'
with importlib_resources.as_file(data_path) as file_path:
    with open(file_path, 'rb') as fp:
        data = fp.read()
assert data.splitlines() == expected_data.splitlines()
expected_data = b'Data entry #2 in `subpkg1/data`.\n'
data_path = pkg_path / 'subpkg1' / 'data' / 'entry2.md'
with importlib_resources.as_file(data_path) as file_path:
    with open(file_path, 'rb') as fp:
        data = fp.read()
assert data.splitlines() == expected_data.splitlines()
data_path = subpkg1_path / 'data' / 'entry2.md'
with importlib_resources.as_file(data_path) as file_path:
    with open(file_path, 'rb') as fp:
        data = fp.read()
assert data.splitlines() == expected_data.splitlines()
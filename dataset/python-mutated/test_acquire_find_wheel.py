from __future__ import annotations
import pytest
from virtualenv.seed.wheels.acquire import find_compatible_in_house
from virtualenv.seed.wheels.embed import BUNDLE_FOLDER, MAX, get_embed_wheel

def test_find_latest_none(for_py_version):
    if False:
        print('Hello World!')
    result = find_compatible_in_house('setuptools', None, for_py_version, BUNDLE_FOLDER)
    expected = get_embed_wheel('setuptools', for_py_version)
    assert result.path == expected.path

def test_find_latest_string(for_py_version):
    if False:
        i = 10
        return i + 15
    result = find_compatible_in_house('setuptools', '', for_py_version, BUNDLE_FOLDER)
    expected = get_embed_wheel('setuptools', for_py_version)
    assert result.path == expected.path

def test_find_exact(for_py_version):
    if False:
        while True:
            i = 10
    expected = get_embed_wheel('setuptools', for_py_version)
    result = find_compatible_in_house('setuptools', f'=={expected.version}', for_py_version, BUNDLE_FOLDER)
    assert result.path == expected.path

def test_find_bad_spec():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='bad'):
        find_compatible_in_house('setuptools', 'bad', MAX, BUNDLE_FOLDER)
""""
Tests scripts/generate-lower-bounds.py
"""
import runpy
import pytest

@pytest.fixture(scope='module')
def script_path(tests_dir):
    if False:
        i = 10
        return i + 15
    return tests_dir.parent / 'scripts' / 'generate-lower-bounds.py'

@pytest.fixture(scope='module')
def generate_lower_bounds(script_path):
    if False:
        print('Hello World!')
    'Retrieves the function that generates lower bounds'
    globals = runpy.run_path(str(script_path))
    return globals['generate_lower_bounds']

def test_generate_lower_bounds_no_version(generate_lower_bounds):
    if False:
        i = 10
        return i + 15
    results = list(generate_lower_bounds(['x']))
    assert results == ['x']

@pytest.mark.parametrize('input', ['x >= 10', 'x >=10', 'x ~=10'])
def test_generate_lower_bounds_min_version_only(generate_lower_bounds, input):
    if False:
        for i in range(10):
            print('nop')
    results = list(generate_lower_bounds([input]))
    assert results == ['x==10']

@pytest.mark.parametrize('min_version', ['10.0', '10.1.3', '10.23.241', '10.0.0.0.0.0'])
def test_generate_lower_bounds_robust_to_versions_with_dots(generate_lower_bounds, min_version):
    if False:
        for i in range(10):
            print('nop')
    results = list(generate_lower_bounds([f'x >= {min_version}']))
    assert results == [f'x=={min_version}']

@pytest.mark.parametrize('min_version', ['10.0a1', '10.0alpha', '10.0a2', '10.0b20', '10.0rc1'])
def test_generate_lower_bounds_robust_to_versions_with_prerelease_designation(generate_lower_bounds, min_version):
    if False:
        print('Hello World!')
    results = list(generate_lower_bounds([f'x >= {min_version}']))
    assert results == [f'x=={min_version}']

@pytest.mark.parametrize('input', ['x <= 11', 'x <=11'])
def test_generate_lower_bounds_max_version_only(generate_lower_bounds, input):
    if False:
        while True:
            i = 10
    results = list(generate_lower_bounds([input]))
    assert results == [input]

@pytest.mark.parametrize('input', ['x != 11', 'x !=11'])
def test_generate_lower_bounds_ignore_version_only(generate_lower_bounds, input):
    if False:
        for i in range(10):
            print('nop')
    results = list(generate_lower_bounds([input]))
    assert results == [input]

@pytest.mark.parametrize('input', ['x <= 12, != 11'])
def test_generate_lower_bounds_ignore_and_max_versions(generate_lower_bounds, input):
    if False:
        print('Hello World!')
    results = list(generate_lower_bounds([input]))
    assert results == [input]

@pytest.mark.parametrize('input', ['x <= 11, >= 10', 'x <=11, >=10', 'x >=10, <=10', 'x >= 10, <= 10'])
def test_generate_lower_bounds_min_and_max_versions(generate_lower_bounds, input):
    if False:
        for i in range(10):
            print('nop')
    results = list(generate_lower_bounds([input]))
    assert results == ['x==10']

@pytest.mark.parametrize('input', ['x != 11, >= 10', 'x !=11, >=10', 'x >=10, !=10', 'x >= 10, != 10'])
def test_generate_lower_bounds_min_and_ignore_versions(generate_lower_bounds, input):
    if False:
        for i in range(10):
            print('nop')
    results = list(generate_lower_bounds([input]))
    assert results == ['x==10']

@pytest.mark.parametrize('input', ['x==10', 'x == 10'])
def test_generate_lower_bounds_pinned_version(generate_lower_bounds, input):
    if False:
        return 10
    results = list(generate_lower_bounds([input]))
    assert results == ['x==10']

@pytest.mark.parametrize('condition', ['python_version < 3.10', 'python_version < 3.10 and foo', 'python_version >= 3'])
def test_generate_lower_bounds_retains_conditions(generate_lower_bounds, condition):
    if False:
        i = 10
        return i + 15
    results = list(generate_lower_bounds([f'x >= 10; {condition}']))
    assert results == [f'x==10; {condition}']
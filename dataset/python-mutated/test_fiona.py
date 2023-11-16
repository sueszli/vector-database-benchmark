import pathlib
import pytest
from pytest_pyodide import run_in_pyodide
TEST_DATA_PATH = pathlib.Path(__file__).parent / 'test_data'

@pytest.mark.driver_timeout(60)
@run_in_pyodide(packages=['fiona'])
def test_supported_drivers(selenium):
    if False:
        while True:
            i = 10
    import fiona
    assert fiona.driver_count() > 0

@pytest.mark.driver_timeout(60)
def test_runtest(selenium):
    if False:
        while True:
            i = 10

    @run_in_pyodide(packages=['fiona', 'pytest'])
    def _run(selenium, data):
        if False:
            for i in range(10):
                print('nop')
        import zipfile
        with open('tests.zip', 'wb') as f:
            f.write(data)
        with zipfile.ZipFile('tests.zip', 'r') as zf:
            zf.extractall('tests')
        import sys
        sys.path.append('tests')
        import pytest

        def runtest(test_filter, ignore_filters):
            if False:
                for i in range(10):
                    print('nop')
            ignore_filter = []
            for ignore in ignore_filters:
                ignore_filter.append('--ignore-glob')
                ignore_filter.append(ignore)
            ret = pytest.main(['--pyargs', 'tests', '--continue-on-collection-errors', *ignore_filter, '-k', test_filter])
            assert ret == 0
        runtest('not ordering and not env and not slice and not GML and not TestNonCountingLayer and not test_schema_default_fields_wrong_type and not http and not FlatGeobuf', ['tests/test_fio*', 'tests/test_data_paths.py', 'tests/test_datetime.py', 'tests/test_vfs.py'])
    TEST_DATA = (TEST_DATA_PATH / 'fiona-tests-1.8.21.zip').read_bytes()
    _run(selenium, TEST_DATA)
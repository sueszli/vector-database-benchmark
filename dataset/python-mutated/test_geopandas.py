import pytest
from pytest_pyodide import run_in_pyodide

@pytest.mark.driver_timeout(300)
@run_in_pyodide(packages=['geopandas', 'geopandas-tests', 'pytest'])
def test_runtest(selenium):
    if False:
        while True:
            i = 10
    from pathlib import Path
    import geopandas
    import pytest
    test_path = Path(geopandas.__file__).parent / 'tests'

    def runtest(test_filter, ignore_filters):
        if False:
            for i in range(10):
                print('nop')
        ignore_filter = []
        for ignore in ignore_filters:
            ignore_filter.append('--ignore-glob')
            ignore_filter.append(ignore)
        ret = pytest.main(['--pyargs', str(test_path), '--continue-on-collection-errors', *ignore_filter, '-k', test_filter])
        assert ret == 0
    runtest('not test_transform2 and not test_no_additional_imports and not test_pandas_kind ', [str(test_path / 'test_dissolve.py'), str(test_path / 'test_geodataframe.py'), str(test_path / 'test_testing.py'), str(test_path / 'test_array.py'), str(test_path / 'test_plotting.py'), str(test_path / 'test_datasets.py'), str(test_path / 'test_extension_array.py'), str(test_path / 'test_crs.py'), str(test_path / 'test_testing.py'), str(test_path / 'test_merge.py'), str(test_path / 'test_explore.py')])
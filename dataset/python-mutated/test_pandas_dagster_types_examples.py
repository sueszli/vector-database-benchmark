import os
import shutil
import pytest
import docs_snippets.guides.dagster.dagster_type_factories as example_root
from dagster import check_dagster_type
from dagster._core.errors import DagsterTypeCheckDidNotPass
from docs_snippets.guides.dagster.dagster_type_factories.job_1 import generate_trip_distribution_plot as job_1
from docs_snippets.guides.dagster.dagster_type_factories.job_2 import generate_trip_distribution_plot as job_2
from docs_snippets.guides.dagster.dagster_type_factories.simple_example import set_containing_1, set_has_element_type_factory
example_root_path = next(iter(example_root.__path__))
EBIKE_TRIPS_PATH = os.path.join(example_root_path, 'ebike_trips.csv')

def test_simple_example_one_off():
    if False:
        return 10
    assert check_dagster_type(set_containing_1, {1, 2}).success

def test_simple_example_factory():
    if False:
        for i in range(10):
            print('nop')
    set_containing_2 = set_has_element_type_factory(2)
    assert check_dagster_type(set_containing_2, {1, 2}).success

@pytest.fixture(scope='function')
def in_tmpdir(monkeypatch, tmp_path_factory):
    if False:
        return 10
    path = tmp_path_factory.mktemp('ebike_trips')
    shutil.copy(EBIKE_TRIPS_PATH, path)
    monkeypatch.chdir(path)

@pytest.mark.usefixtures('in_tmpdir')
def test_job_1_fails():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        job_1.execute_in_process()

@pytest.mark.usefixtures('in_tmpdir')
def test_job_2_no_clean_fails():
    if False:
        return 10
    with pytest.raises(DagsterTypeCheckDidNotPass):
        job_2.execute_in_process()

@pytest.mark.usefixtures('in_tmpdir')
def test_job_2_no_clean_succeeds():
    if False:
        for i in range(10):
            print('nop')
    assert job_2.execute_in_process(run_config={'ops': {'load_trips': {'config': {'clean': True}}}}).success
    assert os.path.exists('./trip_lengths.png')
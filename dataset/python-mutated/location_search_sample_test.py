import re
import backoff
import pytest
import location_search_sample

@pytest.fixture(scope='module')
def company_name():
    if False:
        print('Hello World!')
    (company_name, job_name, job_name2) = location_search_sample.set_up()
    yield company_name
    location_search_sample.tear_down(company_name, job_name, job_name2)

@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_location_search_sample(company_name, capsys):
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, AssertionError, max_time=120)
    def eventually_consistent_test():
        if False:
            return 10
        location_search_sample.run_sample(company_name)
        (out, _) = capsys.readouterr()
        expected = '.*locationFilters.*\n.*locationFilters.*\n.*locationFilters.*\n.*locationFilters.*\n.*locationFilters.*\n'
        assert re.search(expected, out, re.DOTALL)
        expected = '.*matchingJobs.*\n.*matchingJobs.*\n.*matchingJobs.*\n.*matchingJobs.*\n.*matchingJobs.*\n'
        assert re.search(expected, out, re.DOTALL)
    eventually_consistent_test()
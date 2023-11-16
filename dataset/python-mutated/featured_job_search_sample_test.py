import re
import backoff
import pytest
import featured_job_search_sample

@pytest.fixture(scope='module')
def company_name():
    if False:
        while True:
            i = 10
    (company_name, job_name) = featured_job_search_sample.set_up()
    yield company_name
    featured_job_search_sample.tear_down(company_name, job_name)

@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_featured_job_search_sample(company_name, capsys):
    if False:
        for i in range(10):
            print('nop')

    @backoff.on_exception(backoff.expo, AssertionError, max_time=120)
    def eventually_consistent_test():
        if False:
            print('Hello World!')
        featured_job_search_sample.run_sample(company_name)
        (out, _) = capsys.readouterr()
        expected = '.*matchingJobs.*'
        assert re.search(expected, out)
    eventually_consistent_test()
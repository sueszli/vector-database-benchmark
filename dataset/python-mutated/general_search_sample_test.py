import re
import backoff
import pytest
import general_search_sample

@pytest.fixture(scope='module')
def company_and_job():
    if False:
        print('Hello World!')
    (company_name, job_name) = general_search_sample.set_up()
    yield (company_name, job_name)
    general_search_sample.tear_down(company_name, job_name)

@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_general_search_sample(company_and_job, capsys):
    if False:
        return 10

    @backoff.on_exception(backoff.expo, AssertionError, max_time=120)
    def eventually_consistent_test():
        if False:
            return 10
        general_search_sample.run_sample(company_and_job[0], company_and_job[1])
        (out, _) = capsys.readouterr()
        expected = '.*matchingJobs.*\n.*matchingJobs.*\n.*matchingJobs.*\n.*matchingJobs.*\n.*matchingJobs.*\n.*matchingJobs.*\n.*matchingJobs.*\n'
        assert re.search(expected, out, re.DOTALL)
    eventually_consistent_test()
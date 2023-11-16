import re
import backoff
import pytest
import custom_attribute_sample

@pytest.fixture(scope='module')
def create_data():
    if False:
        return 10
    (company_name, job_name) = custom_attribute_sample.set_up()
    yield
    custom_attribute_sample.tear_down(company_name, job_name)

@pytest.mark.flaky(min_passes=1, max_runs=3)
def test_custom_attribute_sample(create_data, capsys):
    if False:
        print('Hello World!')

    @backoff.on_exception(backoff.expo, AssertionError, max_time=120)
    def eventually_consistent_test():
        if False:
            return 10
        custom_attribute_sample.run_sample()
        (out, _) = capsys.readouterr()
        expected = '.*matchingJobs.*job_with_custom_attributes.*\n.*matchingJobs.*job_with_custom_attributes.*\n.*matchingJobs.*job_with_custom_attributes.*\n'
        assert re.search(expected, out, re.DOTALL)
    eventually_consistent_test()
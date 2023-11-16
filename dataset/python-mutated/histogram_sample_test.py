import re
import backoff
import pytest
import histogram_sample

@pytest.fixture(scope='module')
def company_name():
    if False:
        print('Hello World!')
    (company_name, job_name) = histogram_sample.set_up()
    yield company_name
    histogram_sample.tear_down(company_name, job_name)

@pytest.mark.flaky(max_runs=4, min_passes=1)
def test_histogram_sample(company_name, capsys):
    if False:
        return 10

    @backoff.on_exception(backoff.expo, AssertionError, max_time=120)
    def eventually_consistent_test():
        if False:
            return 10
        histogram_sample.run_sample(company_name)
        (out, _) = capsys.readouterr()
        assert re.search('COMPANY_ID', out)
        assert re.search('someFieldName1', out)
    eventually_consistent_test()
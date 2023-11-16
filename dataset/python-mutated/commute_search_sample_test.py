from datetime import date
import re
import backoff
from googleapiclient.errors import HttpError
import pytest
import commute_search_sample

@pytest.fixture(scope='module')
def company_name():
    if False:
        while True:
            i = 10
    (company_name, job_name) = commute_search_sample.set_up()
    yield company_name
    commute_search_sample.tear_down(company_name, job_name)

@pytest.mark.skipif(date.today() < date(2023, 4, 25), reason='Addressed by product team until this date, b/277494438')
@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_commute_search_sample(company_name, capsys):
    if False:
        while True:
            i = 10

    @backoff.on_exception(backoff.expo, (AssertionError, HttpError), max_time=240)
    def eventually_consistent_test():
        if False:
            print('Hello World!')
        commute_search_sample.run_sample(company_name)
        (out, _) = capsys.readouterr()
        expected = '.*matchingJobs.*1600 Amphitheatre Pkwy.*'
        assert re.search(expected, out)
    eventually_consistent_test()
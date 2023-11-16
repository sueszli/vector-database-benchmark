import re
import backoff
import pytest
import auto_complete_sample

@pytest.fixture(scope='module')
def company_name():
    if False:
        return 10
    (company_name, job_name) = auto_complete_sample.set_up()
    yield company_name
    auto_complete_sample.tear_down(company_name, job_name)

def test_auto_complete_sample(company_name, capsys):
    if False:
        return 10

    @backoff.on_exception(backoff.expo, AssertionError, max_time=120)
    def eventually_consistent_test():
        if False:
            while True:
                i = 10
        auto_complete_sample.run_sample(company_name)
        (out, _) = capsys.readouterr()
        expected = '.*completionResults.*suggestion.*Google.*type.*COMPANY_NAME.*\n.*completionResults.*suggestion.*Software Engineer.*type.*JOB_TITLE.*\n.*completionResults.*suggestion.*Software Engineer.*type.*JOB_TITLE.*\n'
        assert re.search(expected, out)
    eventually_consistent_test()
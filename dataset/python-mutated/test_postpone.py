import datetime
import logging
import pytest
try:
    import dateutil.parser
    HAS_DATEUTIL_PARSER = True
except ImportError:
    HAS_DATEUTIL_PARSER = False
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.skipif(HAS_DATEUTIL_PARSER is False, reason="The 'dateutil.parser' library is not available"), pytest.mark.windows_whitelisted]

@pytest.mark.slow_test
def test_postpone(schedule):
    if False:
        print('Hello World!')
    '\n    verify that scheduled job is postponed until the specified time.\n    '
    job = {'schedule': {'job1': {'function': 'test.ping', 'when': '11/29/2017 4pm'}}}
    run_time = dateutil.parser.parse('11/29/2017 4:00pm')
    delay = 300
    schedule.opts.update(job)
    schedule.postpone_job('job1', {'time': run_time.strftime('%Y-%m-%dT%H:%M:%S'), 'new_time': (run_time + datetime.timedelta(seconds=delay)).strftime('%Y-%m-%dT%H:%M:%S')})
    schedule.eval(now=run_time)
    ret = schedule.job_status('job1')
    assert '_last_run' not in ret
    schedule.eval(now=run_time + datetime.timedelta(seconds=delay))
    ret = schedule.job_status('job1')
    assert ret['_last_run'] == run_time + datetime.timedelta(seconds=delay)
    schedule.eval(now=run_time + datetime.timedelta(seconds=delay + 1))
    ret = schedule.job_status('job1')
    assert ret['_last_run'] == run_time + datetime.timedelta(seconds=delay)
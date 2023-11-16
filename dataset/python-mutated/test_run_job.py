import logging
log = logging.getLogger(__name__)

def test_run_job(schedule):
    if False:
        print('Hello World!')
    '\n    verify that scheduled job runs\n    '
    job_name = 'test_run_job'
    job = {'schedule': {job_name: {'function': 'test.ping'}}}
    schedule.opts.update(job)
    schedule.run_job(job_name)
    ret = schedule.job_status(job_name)
    expected = {'function': 'test.ping', 'run': True, 'name': 'test_run_job'}
    assert ret == expected
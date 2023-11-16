import time
from rq import Queue, SimpleWorker
from rq.registry import FailedJobRegistry, FinishedJobRegistry
from rq.timeouts import TimerDeathPenalty
from tests import RQTestCase

class TimerBasedWorker(SimpleWorker):
    death_penalty_class = TimerDeathPenalty

def thread_friendly_sleep_func(seconds):
    if False:
        while True:
            i = 10
    end_at = time.time() + seconds
    while True:
        if time.time() > end_at:
            break

class TestTimeouts(RQTestCase):

    def test_timer_death_penalty(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure TimerDeathPenalty works correctly.'
        q = Queue(connection=self.testconn)
        q.empty()
        finished_job_registry = FinishedJobRegistry(connection=self.testconn)
        failed_job_registry = FailedJobRegistry(connection=self.testconn)
        w = TimerBasedWorker([q], connection=self.testconn)
        self.assertIsNotNone(w)
        self.assertEqual(w.death_penalty_class, TimerDeathPenalty)
        job = q.enqueue(thread_friendly_sleep_func, args=(1,), job_timeout=3)
        w.work(burst=True)
        job.refresh()
        self.assertIn(job, finished_job_registry)
        job = q.enqueue(thread_friendly_sleep_func, args=(5,), job_timeout=3)
        w.work(burst=True)
        self.assertIn(job, failed_job_registry)
        job.refresh()
        self.assertIn('rq.timeouts.JobTimeoutException', job.exc_info)
        job = q.enqueue(thread_friendly_sleep_func, args=(1,), job_timeout=-1)
        w.work(burst=True)
        job.refresh()
        self.assertIn(job, finished_job_registry)
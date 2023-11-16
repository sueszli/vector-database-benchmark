import time
from contextlib import contextmanager
from unittest.mock import patch
from rq import Queue
import frappe
from frappe.core.doctype.rq_job.rq_job import remove_failed_jobs
from frappe.tests.utils import FrappeTestCase
from frappe.utils.background_jobs import RQ_JOB_FAILURE_TTL, RQ_RESULTS_TTL, create_job_id, execute_job, generate_qname, get_redis_conn

class TestBackgroundJobs(FrappeTestCase):

    def test_remove_failed_jobs(self):
        if False:
            i = 10
            return i + 15
        frappe.enqueue(method='frappe.tests.test_background_jobs.fail_function', queue='short')
        time.sleep(2)
        conn = get_redis_conn()
        queues = Queue.all(conn)
        for queue in queues:
            if queue.name == generate_qname('short'):
                fail_registry = queue.failed_job_registry
                self.assertGreater(fail_registry.count, 0)
        remove_failed_jobs()
        for queue in queues:
            if queue.name == generate_qname('short'):
                fail_registry = queue.failed_job_registry
                self.assertEqual(fail_registry.count, 0)

    def test_enqueue_at_front(self):
        if False:
            return 10
        kwargs = {'method': 'frappe.handler.ping', 'queue': 'short'}
        frappe.enqueue(**kwargs)
        low_priority_job = frappe.enqueue(**kwargs)
        high_priority_job = frappe.enqueue(**kwargs, at_front=True)
        self.assertTrue(high_priority_job.get_position() < low_priority_job.get_position())

    def test_job_hooks(self):
        if False:
            return 10
        self.addCleanup(lambda : _test_JOB_HOOK.clear())
        with freeze_local() as locals, frappe.init_site(locals.site), patch('frappe.get_hooks', patch_job_hooks):
            frappe.connect()
            self.assertIsNone(_test_JOB_HOOK.get('before_job'))
            r = execute_job(site=frappe.local.site, user='Administrator', method='frappe.handler.ping', event=None, job_name='frappe.handler.ping', is_async=True, kwargs={})
            self.assertEqual(r, 'pong')
            self.assertLess(_test_JOB_HOOK.get('before_job'), _test_JOB_HOOK.get('after_job'))

def fail_function():
    if False:
        print('Hello World!')
    return 1 / 0
_test_JOB_HOOK = {}

def before_job(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    _test_JOB_HOOK['before_job'] = time.time()

def after_job(*args, **kwargs):
    if False:
        return 10
    _test_JOB_HOOK['after_job'] = time.time()

@contextmanager
def freeze_local():
    if False:
        print('Hello World!')
    locals = frappe.local
    frappe.local = frappe.Local()
    yield locals
    frappe.local = locals

def patch_job_hooks(event: str):
    if False:
        for i in range(10):
            print('nop')
    return {'before_job': ['frappe.tests.test_background_jobs.before_job'], 'after_job': ['frappe.tests.test_background_jobs.after_job']}[event]
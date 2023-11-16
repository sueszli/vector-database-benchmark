import frappe
from frappe.core.doctype.rq_worker.rq_worker import RQWorker
from frappe.tests.utils import FrappeTestCase

class TestRQWorker(FrappeTestCase):

    def test_get_worker_list(self):
        if False:
            return 10
        workers = RQWorker.get_list({})
        self.assertGreaterEqual(len(workers), 1)
        self.assertTrue(any(('short' in w.queue_type for w in workers)))

    def test_worker_serialization(self):
        if False:
            for i in range(10):
                print('nop')
        workers = RQWorker.get_list({})
        frappe.get_doc('RQ Worker', workers[0].pid)
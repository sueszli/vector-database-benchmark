from frappe.model.docstatus import DocStatus
from frappe.tests.utils import FrappeTestCase

class TestDocStatus(FrappeTestCase):

    def test_draft(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(DocStatus(0), DocStatus.draft())
        self.assertTrue(DocStatus.draft().is_draft())
        self.assertFalse(DocStatus.draft().is_cancelled())
        self.assertFalse(DocStatus.draft().is_submitted())

    def test_submitted(self):
        if False:
            return 10
        self.assertEqual(DocStatus(1), DocStatus.submitted())
        self.assertFalse(DocStatus.submitted().is_draft())
        self.assertTrue(DocStatus.submitted().is_submitted())
        self.assertFalse(DocStatus.submitted().is_cancelled())

    def test_cancelled(self):
        if False:
            while True:
                i = 10
        self.assertEqual(DocStatus(2), DocStatus.cancelled())
        self.assertFalse(DocStatus.cancelled().is_draft())
        self.assertFalse(DocStatus.cancelled().is_submitted())
        self.assertTrue(DocStatus.cancelled().is_cancelled())
import frappe
from frappe.core.doctype.doctype.test_doctype import new_doctype
from frappe.tests.utils import FrappeTestCase

class TestRating(FrappeTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        doc = new_doctype(fields=[{'fieldname': 'rating', 'fieldtype': 'Rating', 'label': 'rating', 'reqd': 1}])
        doc.insert()
        self.doctype_name = doc.name

    def test_negative_rating(self):
        if False:
            while True:
                i = 10
        doc = frappe.new_doc(doctype=self.doctype_name, rating=-1)
        doc.insert()
        self.assertEqual(doc.rating, 0)

    def test_positive_rating(self):
        if False:
            print('Hello World!')
        doc = frappe.new_doc(doctype=self.doctype_name, rating=5)
        doc.insert()
        self.assertEqual(doc.rating, 1)
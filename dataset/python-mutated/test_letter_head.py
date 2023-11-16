import frappe
from frappe.tests.utils import FrappeTestCase

class TestLetterHead(FrappeTestCase):

    def test_auto_image(self):
        if False:
            i = 10
            return i + 15
        letter_head = frappe.get_doc(dict(doctype='Letter Head', letter_head_name='Test', source='Image', image='/public/test.png')).insert()
        self.assertTrue(letter_head.image in letter_head.content)
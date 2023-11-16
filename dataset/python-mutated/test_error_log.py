from unittest.mock import patch
from ldap3.core.exceptions import LDAPException, LDAPInappropriateAuthenticationResult
import frappe
from frappe.tests.utils import FrappeTestCase
from frappe.utils.error import _is_ldap_exception, guess_exception_source

class TestErrorLog(FrappeTestCase):

    def test_error_log(self):
        if False:
            i = 10
            return i + 15
        "let's do an error log on error log?"
        doc = frappe.new_doc('Error Log')
        error = doc.log_error('This is an error')
        self.assertEqual(error.doctype, 'Error Log')

    def test_ldap_exceptions(self):
        if False:
            print('Hello World!')
        exc = [LDAPException, LDAPInappropriateAuthenticationResult]
        for e in exc:
            self.assertTrue(_is_ldap_exception(e()))
_RAW_EXC = '\n   File "apps/frappe/frappe/model/document.py", line 1284, in runner\n     add_to_return_value(self, fn(self, *args, **kwargs))\n                               ^^^^^^^^^^^^^^^^^^^^^^^^^\n   File "apps/frappe/frappe/model/document.py", line 933, in fn\n     return method_object(*args, **kwargs)\n            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n   File "apps/erpnext/erpnext/selling/doctype/sales_order/sales_order.py", line 58, in onload\n     raise Exception("what")\n Exception: what\n'
_THROW_EXC = '\n   File "apps/frappe/frappe/model/document.py", line 933, in fn\n     return method_object(*args, **kwargs)\n            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n   File "apps/erpnext/erpnext/selling/doctype/sales_order/sales_order.py", line 58, in onload\n     frappe.throw("what")\n   File "apps/frappe/frappe/__init__.py", line 550, in throw\n     msgprint(\n   File "apps/frappe/frappe/__init__.py", line 518, in msgprint\n     _raise_exception()\n   File "apps/frappe/frappe/__init__.py", line 467, in _raise_exception\n     raise raise_exception(msg)\n frappe.exceptions.ValidationError: what\n'
TEST_EXCEPTIONS = {'erpnext (app)': _RAW_EXC, 'erpnext (app)': _THROW_EXC}

class TestExceptionSourceGuessing(FrappeTestCase):

    @patch.object(frappe, 'get_installed_apps', return_value=['frappe', 'erpnext', '3pa'])
    def test_exc_source_guessing(self, _installed_apps):
        if False:
            print('Hello World!')
        for (source, exc) in TEST_EXCEPTIONS.items():
            result = guess_exception_source(exc)
            self.assertEqual(result, source)
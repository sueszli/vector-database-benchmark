import base64
import os
import requests
import frappe
from frappe.core.doctype.access_log.access_log import make_access_log
from frappe.core.doctype.data_import.data_import import export_csv
from frappe.core.doctype.user.user import generate_keys
from frappe.tests.utils import FrappeTestCase
from frappe.utils import cstr, get_site_url

class TestAccessLog(FrappeTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        generate_keys(frappe.session.user)
        frappe.db.commit()
        generated_secret = frappe.utils.password.get_decrypted_password('User', frappe.session.user, fieldname='api_secret')
        api_key = frappe.db.get_value('User', 'Administrator', 'api_key')
        self.header = {'Authorization': f'token {api_key}:{generated_secret}'}
        self.test_html_template = '\n\t\t\t<!DOCTYPE html>\n\t\t\t<html>\n\t\t\t<head>\n\t\t\t<style>\n\t\t\ttable {\n\t\t\tfont-family: arial, sans-serif;\n\t\t\tborder-collapse: collapse;\n\t\t\twidth: 100%;\n\t\t\t}\n\n\t\t\ttd, th {\n\t\t\tborder: 1px solid #dddddd;\n\t\t\ttext-align: left;\n\t\t\tpadding: 8px;\n\t\t\t}\n\n\t\t\ttr:nth-child(even) {\n\t\t\tbackground-color: #dddddd;\n\t\t\t}\n\t\t\t</style>\n\t\t\t</head>\n\t\t\t<body>\n\n\t\t\t<h2>HTML Table</h2>\n\n\t\t\t<table>\n\t\t\t<tr>\n\t\t\t\t<th>Company</th>\n\t\t\t\t<th>Contact</th>\n\t\t\t\t<th>Country</th>\n\t\t\t</tr>\n\t\t\t<tr>\n\t\t\t\t<td>Alfreds Futterkiste</td>\n\t\t\t\t<td>Maria Anders</td>\n\t\t\t\t<td>Germany</td>\n\t\t\t</tr>\n\t\t\t<tr>\n\t\t\t\t<td>Centro comercial Moctezuma</td>\n\t\t\t\t<td>Francisco Chang</td>\n\t\t\t\t<td>Mexico</td>\n\t\t\t</tr>\n\t\t\t<tr>\n\t\t\t\t<td>Ernst Handel</td>\n\t\t\t\t<td>Roland Mendel</td>\n\t\t\t\t<td>Austria</td>\n\t\t\t</tr>\n\t\t\t<tr>\n\t\t\t\t<td>Island Trading</td>\n\t\t\t\t<td>Helen Bennett</td>\n\t\t\t\t<td>UK</td>\n\t\t\t</tr>\n\t\t\t<tr>\n\t\t\t\t<td>Laughing Bacchus Winecellars</td>\n\t\t\t\t<td>Yoshi Tannamuri</td>\n\t\t\t\t<td>Canada</td>\n\t\t\t</tr>\n\t\t\t<tr>\n\t\t\t\t<td>Magazzini Alimentari Riuniti</td>\n\t\t\t\t<td>Giovanni Rovelli</td>\n\t\t\t\t<td>Italy</td>\n\t\t\t</tr>\n\t\t\t</table>\n\n\t\t\t</body>\n\t\t\t</html>\n\t\t'
        self.test_filters = {'from_date': '2019-06-30', 'to_date': '2019-07-31', 'party': [], 'group_by': 'Group by Voucher (Consolidated)', 'cost_center': [], 'project': []}
        self.test_doctype = 'File'
        self.test_document = 'Test Document'
        self.test_report_name = 'General Ledger'
        self.test_file_type = 'CSV'
        self.test_method = 'Test Method'
        self.file_name = frappe.utils.random_string(10) + '.txt'
        self.test_content = frappe.utils.random_string(1024)

    def test_make_full_access_log(self):
        if False:
            print('Hello World!')
        self.maxDiff = None
        make_access_log(doctype=self.test_doctype, document=self.test_document, report_name=self.test_report_name, page=self.test_html_template, file_type=self.test_file_type, method=self.test_method, filters=self.test_filters)
        last_doc = frappe.get_last_doc('Access Log')
        self.assertEqual(last_doc.filters, cstr(self.test_filters))
        self.assertEqual(self.test_doctype, last_doc.export_from)
        self.assertEqual(self.test_document, last_doc.reference_document)

    def test_make_export_log(self):
        if False:
            while True:
                i = 10
        export_csv(self.test_doctype, self.file_name)
        os.remove(self.file_name)
        last_doc = frappe.get_last_doc('Access Log')
        self.assertEqual(self.test_doctype, last_doc.export_from)

    def test_private_file_download(self):
        if False:
            while True:
                i = 10
        new_private_file = frappe.get_doc({'doctype': self.test_doctype, 'file_name': self.file_name, 'content': base64.b64encode(self.test_content.encode('utf-8')), 'is_private': 1})
        new_private_file.insert()
        private_file_link = get_site_url(frappe.local.site) + new_private_file.file_url
        try:
            request = requests.post(private_file_link, headers=self.header)
            last_doc = frappe.get_last_doc('Access Log')
            if request.ok:
                self.assertEqual(new_private_file.doctype, last_doc.export_from)
                self.assertEqual(new_private_file.name, last_doc.reference_document)
        except requests.ConnectionError:
            pass
        new_private_file.delete()

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass
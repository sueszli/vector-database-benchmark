import os
from unittest.mock import patch
import frappe
import frappe.utils
from frappe.core.doctype.doctype.test_doctype import new_doctype
from frappe.desk.notifications import get_open_count
from frappe.tests.utils import FrappeTestCase

class TestDashboardConnections(FrappeTestCase):

    @patch.dict(frappe.conf, {'developer_mode': 1})
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        delete_test_data()
        create_test_data()

    @patch.dict(frappe.conf, {'developer_mode': 1})
    def tearDown(self):
        if False:
            while True:
                i = 10
        delete_test_data()

    def test_internal_link_count(self):
        if False:
            print('Hello World!')
        earth = frappe.get_doc({'doctype': 'Test Doctype B With Child Table With Link To Doctype A', 'title': 'Earth'})
        earth.append('child_table', {'title': 'Earth'})
        earth.insert()
        mars = frappe.get_doc({'doctype': 'Test Doctype A With Child Table With Link To Doctype B', 'title': 'Mars'})
        mars.append('child_table', {'title': 'Mars', 'test_doctype_b_with_test_child_table_with_link_to_doctype_a': 'Earth'})
        mars.insert()
        expected_open_count = {'count': {'external_links_found': [], 'internal_links_found': [{'count': 1, 'doctype': 'Test Doctype B With Child Table With Link To Doctype A', 'names': ['Earth'], 'open_count': 0}]}}
        with patch.object(mars.meta, 'get_dashboard_data', return_value=get_dashboard_for_test_doctype_a_with_test_child_table_with_link_to_doctype_b()):
            self.assertEqual(get_open_count('Test Doctype A With Child Table With Link To Doctype B', 'Mars'), expected_open_count)

    def test_external_link_count(self):
        if False:
            while True:
                i = 10
        saturn = frappe.get_doc({'doctype': 'Test Doctype A With Child Table With Link To Doctype B', 'title': 'Saturn'})
        saturn.append('child_table', {'title': 'Saturn'})
        saturn.insert()
        pluto = frappe.get_doc({'doctype': 'Test Doctype B With Child Table With Link To Doctype A', 'title': 'Pluto'})
        pluto.append('child_table', {'title': 'Pluto', 'test_doctype_a_with_test_child_table_with_link_to_doctype_b': 'Saturn'})
        pluto.insert()
        expected_open_count = {'count': {'external_links_found': [{'doctype': 'Test Doctype B With Child Table With Link To Doctype A', 'open_count': 0, 'count': 1}], 'internal_links_found': []}}
        with patch.object(saturn.meta, 'get_dashboard_data', return_value=get_dashboard_for_test_doctype_a_with_test_child_table_with_link_to_doctype_b()):
            self.assertEqual(get_open_count('Test Doctype A With Child Table With Link To Doctype B', 'Saturn'), expected_open_count)

def create_test_data():
    if False:
        return 10
    create_test_child_table_with_link_to_doctype_a()
    create_test_child_table_with_link_to_doctype_b()
    create_test_doctype_a_with_test_child_table_with_link_to_doctype_b()
    create_test_doctype_b_with_test_child_table_with_link_to_doctype_a()
    add_links_in_child_tables()

def delete_test_data():
    if False:
        while True:
            i = 10
    doctypes = ['Test Child Table With Link To Doctype A', 'Test Child Table With Link To Doctype B', 'Test Doctype A With Child Table With Link To Doctype B', 'Test Doctype B With Child Table With Link To Doctype A']
    for doctype in doctypes:
        if frappe.db.table_exists(doctype):
            frappe.db.delete(doctype)
            frappe.delete_doc('DocType', doctype, force=True)

def create_test_child_table_with_link_to_doctype_a():
    if False:
        while True:
            i = 10
    new_doctype('Test Child Table With Link To Doctype A', istable=1, fields=[{'fieldname': 'title', 'fieldtype': 'Data', 'label': 'Title', 'reqd': 1, 'unique': 1}], custom=False, autoname='field:title', naming_rule='By fieldname').insert(ignore_if_duplicate=True)

def create_test_child_table_with_link_to_doctype_b():
    if False:
        for i in range(10):
            print('nop')
    new_doctype('Test Child Table With Link To Doctype B', istable=1, fields=[{'fieldname': 'title', 'fieldtype': 'Data', 'label': 'Title', 'reqd': 1, 'unique': 1}], custom=False, autoname='field:title', naming_rule='By fieldname').insert(ignore_if_duplicate=True)

def add_links_in_child_tables():
    if False:
        for i in range(10):
            print('nop')
    test_child_table_with_link_to_doctype_a = frappe.get_doc('DocType', 'Test Child Table With Link To Doctype A')
    if len(test_child_table_with_link_to_doctype_a.fields) == 1:
        test_child_table_with_link_to_doctype_a.append('fields', {'fieldname': 'test_doctype_a_with_test_child_table_with_link_to_doctype_b', 'fieldtype': 'Link', 'in_list_view': 1, 'label': 'Test Doctype A With Child Table With Link To Doctype B' or 'Doctype to Link', 'options': 'Test Doctype A With Child Table With Link To Doctype B' or 'Doctype to Link'})
        test_child_table_with_link_to_doctype_a.save()
    test_child_table_with_link_to_doctype_b = frappe.get_doc('DocType', 'Test Child Table With Link To Doctype B')
    if len(test_child_table_with_link_to_doctype_b.fields) == 1:
        test_child_table_with_link_to_doctype_b.append('fields', {'fieldname': 'test_doctype_b_with_test_child_table_with_link_to_doctype_a', 'fieldtype': 'Link', 'in_list_view': 1, 'label': 'Test Doctype B With Child Table With Link To Doctype A' or 'Doctype to Link', 'options': 'Test Doctype B With Child Table With Link To Doctype A' or 'Doctype to Link'})
        test_child_table_with_link_to_doctype_b.save()

def create_test_doctype_a_with_test_child_table_with_link_to_doctype_b():
    if False:
        i = 10
        return i + 15
    new_doctype('Test Doctype A With Child Table With Link To Doctype B', fields=[{'fieldname': 'title', 'fieldtype': 'Data', 'label': 'Title', 'unique': 1}, {'fieldname': 'child_table', 'fieldtype': 'Table', 'label': 'Child Table', 'options': 'Test Child Table With Link To Doctype B'}, {'fieldname': 'connections_tab', 'fieldtype': 'Tab Break', 'label': 'Connections', 'show_dashboard': 1}], custom=False, autoname='field:title', naming_rule='By fieldname').insert(ignore_if_duplicate=True)

def create_test_doctype_b_with_test_child_table_with_link_to_doctype_a():
    if False:
        while True:
            i = 10
    new_doctype('Test Doctype B With Child Table With Link To Doctype A', fields=[{'fieldname': 'title', 'fieldtype': 'Data', 'label': 'Title', 'unique': 1}, {'fieldname': 'child_table', 'fieldtype': 'Table', 'label': 'Child Table', 'options': 'Test Child Table With Link To Doctype A'}, {'fieldname': 'connections_tab', 'fieldtype': 'Tab Break', 'label': 'Connections', 'show_dashboard': 1}], custom=False, autoname='field:title', naming_rule='By fieldname').insert(ignore_if_duplicate=True)

def get_dashboard_for_test_doctype_a_with_test_child_table_with_link_to_doctype_b():
    if False:
        while True:
            i = 10
    dashboard = frappe._dict()
    data = {'fieldname': 'test_doctype_a_with_test_child_table_with_link_to_doctype_b', 'internal_and_external_links': {'Test Doctype B With Child Table With Link To Doctype A': ['child_table', 'test_doctype_b_with_test_child_table_with_link_to_doctype_a']}, 'transactions': [{'label': 'Reference', 'items': ['Test Doctype B With Child Table With Link To Doctype A']}]}
    dashboard.fieldname = data['fieldname']
    dashboard.internal_and_external_links = data['internal_and_external_links']
    dashboard.transactions = data['transactions']
    return dashboard
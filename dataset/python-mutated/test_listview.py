import json
import frappe
from frappe.desk.listview import get_group_by_count, get_list_settings, set_list_settings
from frappe.desk.reportview import get
from frappe.tests.utils import FrappeTestCase

class TestListView(FrappeTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        if frappe.db.exists('List View Settings', 'DocType'):
            frappe.delete_doc('List View Settings', 'DocType')

    def test_get_list_settings_without_settings(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNone(get_list_settings('DocType'), None)

    def test_get_list_settings_with_default_settings(self):
        if False:
            print('Hello World!')
        frappe.get_doc({'doctype': 'List View Settings', 'name': 'DocType'}).insert()
        settings = get_list_settings('DocType')
        self.assertIsNotNone(settings)
        self.assertEqual(settings.disable_auto_refresh, 0)
        self.assertEqual(settings.disable_count, 0)
        self.assertEqual(settings.disable_comment_count, 0)
        self.assertEqual(settings.disable_sidebar_stats, 0)

    def test_get_list_settings_with_non_default_settings(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.get_doc({'doctype': 'List View Settings', 'name': 'DocType', 'disable_count': 1}).insert()
        settings = get_list_settings('DocType')
        self.assertIsNotNone(settings)
        self.assertEqual(settings.disable_auto_refresh, 0)
        self.assertEqual(settings.disable_count, 1)
        self.assertEqual(settings.disable_comment_count, 0)
        self.assertEqual(settings.disable_sidebar_stats, 0)

    def test_set_list_settings_without_settings(self):
        if False:
            for i in range(10):
                print('nop')
        set_list_settings('DocType', json.dumps({}))
        settings = frappe.get_doc('List View Settings', 'DocType')
        self.assertEqual(settings.disable_auto_refresh, 0)
        self.assertEqual(settings.disable_count, 0)
        self.assertEqual(settings.disable_comment_count, 0)
        self.assertEqual(settings.disable_sidebar_stats, 0)

    def test_set_list_settings_with_existing_settings(self):
        if False:
            print('Hello World!')
        frappe.get_doc({'doctype': 'List View Settings', 'name': 'DocType', 'disable_count': 1}).insert()
        set_list_settings('DocType', json.dumps({'disable_count': 0, 'disable_auto_refresh': 1}))
        settings = frappe.get_doc('List View Settings', 'DocType')
        self.assertEqual(settings.disable_auto_refresh, 1)
        self.assertEqual(settings.disable_count, 0)
        self.assertEqual(settings.disable_comment_count, 0)
        self.assertEqual(settings.disable_sidebar_stats, 0)

    def test_list_view_child_table_filter_with_created_by_filter(self):
        if False:
            for i in range(10):
                print('nop')
        if frappe.db.exists('Note', 'Test created by filter with child table filter'):
            frappe.delete_doc('Note', 'Test created by filter with child table filter')
        doc = frappe.get_doc({'doctype': 'Note', 'title': 'Test created by filter with child table filter', 'public': 1})
        doc.append('seen_by', {'user': 'Administrator'})
        doc.insert()
        data = {d.name: d.count for d in get_group_by_count('Note', '[["Note Seen By","user","=","Administrator"]]', 'owner')}
        self.assertEqual(data['Administrator'], 1)

    def test_get_group_by_invalid_field(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, get_group_by_count, 'Note', '[["Note Seen By","user","=","Administrator"]]', 'invalid_field')

    def test_list_view_comment_count(self):
        if False:
            i = 10
            return i + 15
        frappe.form_dict.doctype = 'DocType'
        frappe.form_dict.limit = '1'
        frappe.form_dict.fields = ['`tabDocType`.`name`']
        for with_comment_count in (1, True, '1'):
            frappe.form_dict.with_comment_count = with_comment_count
            self.assertEqual(len(get()['values'][0]), 2)
        for with_comment_count in (0, False, '0', None):
            frappe.form_dict.with_comment_count = with_comment_count
            self.assertEqual(len(get()['values'][0]), 1)
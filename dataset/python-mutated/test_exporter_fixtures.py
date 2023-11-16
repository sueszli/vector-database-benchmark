import os
import frappe
import frappe.defaults
from frappe.core.doctype.data_import.data_import import export_csv
from frappe.tests.utils import FrappeTestCase

class TestDataImportFixtures(FrappeTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_Custom_Script_fixture_simple(self):
        if False:
            for i in range(10):
                print('nop')
        fixture = 'Client Script'
        path = frappe.scrub(fixture) + '_original_style.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_simple_name_equal_default(self):
        if False:
            return 10
        fixture = ['Client Script', {'name': ['Item']}]
        path = frappe.scrub(fixture[0]) + '_simple_name_equal_default.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_simple_name_equal(self):
        if False:
            while True:
                i = 10
        fixture = ['Client Script', {'name': ['Item'], 'op': '='}]
        path = frappe.scrub(fixture[0]) + '_simple_name_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_simple_name_not_equal(self):
        if False:
            return 10
        fixture = ['Client Script', {'name': ['Item'], 'op': '!='}]
        path = frappe.scrub(fixture[0]) + '_simple_name_not_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_simple_name_at_least_equal(self):
        if False:
            return 10
        fixture = ['Client Script', {'name': 'Item-Cli'}]
        path = frappe.scrub(fixture[0]) + '_simple_name_at_least_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_multi_name_equal(self):
        if False:
            while True:
                i = 10
        fixture = ['Client Script', {'name': ['Item', 'Customer'], 'op': '='}]
        path = frappe.scrub(fixture[0]) + '_multi_name_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_multi_name_not_equal(self):
        if False:
            i = 10
            return i + 15
        fixture = ['Client Script', {'name': ['Item', 'Customer'], 'op': '!='}]
        path = frappe.scrub(fixture[0]) + '_multi_name_not_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_empty_object(self):
        if False:
            print('Hello World!')
        fixture = ['Client Script', {}]
        path = frappe.scrub(fixture[0]) + '_empty_object_should_be_all.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_just_list(self):
        if False:
            print('Hello World!')
        fixture = ['Client Script']
        path = frappe.scrub(fixture[0]) + '_just_list_should_be_all.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_rex_no_flags(self):
        if False:
            i = 10
            return i + 15
        fixture = ['Client Script', {'name': '^[i|A]'}]
        path = frappe.scrub(fixture[0]) + '_rex_no_flags.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Script_fixture_rex_with_flags(self):
        if False:
            i = 10
            return i + 15
        fixture = ['Client Script', {'name': '^[i|A]', 'flags': 'L,M'}]
        path = frappe.scrub(fixture[0]) + '_rex_with_flags.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_simple(self):
        if False:
            return 10
        fixture = 'Custom Field'
        path = frappe.scrub(fixture) + '_original_style.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_simple_name_equal_default(self):
        if False:
            i = 10
            return i + 15
        fixture = ['Custom Field', {'name': ['Item-vat']}]
        path = frappe.scrub(fixture[0]) + '_simple_name_equal_default.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_simple_name_equal(self):
        if False:
            return 10
        fixture = ['Custom Field', {'name': ['Item-vat'], 'op': '='}]
        path = frappe.scrub(fixture[0]) + '_simple_name_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_simple_name_not_equal(self):
        if False:
            print('Hello World!')
        fixture = ['Custom Field', {'name': ['Item-vat'], 'op': '!='}]
        path = frappe.scrub(fixture[0]) + '_simple_name_not_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_simple_name_at_least_equal(self):
        if False:
            for i in range(10):
                print('nop')
        fixture = ['Custom Field', {'name': 'Item-va'}]
        path = frappe.scrub(fixture[0]) + '_simple_name_at_least_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_multi_name_equal(self):
        if False:
            print('Hello World!')
        fixture = ['Custom Field', {'name': ['Item-vat', 'Bin-vat'], 'op': '='}]
        path = frappe.scrub(fixture[0]) + '_multi_name_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_multi_name_not_equal(self):
        if False:
            return 10
        fixture = ['Custom Field', {'name': ['Item-vat', 'Bin-vat'], 'op': '!='}]
        path = frappe.scrub(fixture[0]) + '_multi_name_not_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_empty_object(self):
        if False:
            return 10
        fixture = ['Custom Field', {}]
        path = frappe.scrub(fixture[0]) + '_empty_object_should_be_all.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_just_list(self):
        if False:
            for i in range(10):
                print('nop')
        fixture = ['Custom Field']
        path = frappe.scrub(fixture[0]) + '_just_list_should_be_all.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_rex_no_flags(self):
        if False:
            i = 10
            return i + 15
        fixture = ['Custom Field', {'name': '^[r|L]'}]
        path = frappe.scrub(fixture[0]) + '_rex_no_flags.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Custom_Field_fixture_rex_with_flags(self):
        if False:
            for i in range(10):
                print('nop')
        fixture = ['Custom Field', {'name': '^[i|A]', 'flags': 'L,M'}]
        path = frappe.scrub(fixture[0]) + '_rex_with_flags.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_fixture_simple(self):
        if False:
            return 10
        fixture = 'ToDo'
        path = 'Doctype_' + frappe.scrub(fixture) + '_original_style_should_be_all.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_fixture_simple_name_equal_default(self):
        if False:
            return 10
        fixture = ['ToDo', {'name': ['TDI00000008']}]
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_simple_name_equal_default.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_fixture_simple_name_equal(self):
        if False:
            for i in range(10):
                print('nop')
        fixture = ['ToDo', {'name': ['TDI00000002'], 'op': '='}]
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_simple_name_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_simple_name_not_equal(self):
        if False:
            for i in range(10):
                print('nop')
        fixture = ['ToDo', {'name': ['TDI00000002'], 'op': '!='}]
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_simple_name_not_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_fixture_simple_name_at_least_equal(self):
        if False:
            i = 10
            return i + 15
        fixture = ['ToDo', {'name': 'TDI'}]
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_simple_name_at_least_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_multi_name_equal(self):
        if False:
            i = 10
            return i + 15
        fixture = ['ToDo', {'name': ['TDI00000002', 'TDI00000008'], 'op': '='}]
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_multi_name_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_multi_name_not_equal(self):
        if False:
            while True:
                i = 10
        fixture = ['ToDo', {'name': ['TDI00000002', 'TDI00000008'], 'op': '!='}]
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_multi_name_not_equal.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_fixture_empty_object(self):
        if False:
            for i in range(10):
                print('nop')
        fixture = ['ToDo', {}]
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_empty_object_should_be_all.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_fixture_just_list(self):
        if False:
            print('Hello World!')
        fixture = ['ToDo']
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_just_list_should_be_all.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_fixture_rex_no_flags(self):
        if False:
            return 10
        fixture = ['ToDo', {'name': '^TDi'}]
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_rex_no_flags_should_be_all.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)

    def test_Doctype_fixture_rex_with_flags(self):
        if False:
            while True:
                i = 10
        fixture = ['ToDo', {'name': '^TDi', 'flags': 'L,M'}]
        path = 'Doctype_' + frappe.scrub(fixture[0]) + '_rex_with_flags_should_be_none.csv'
        export_csv(fixture, path)
        self.assertTrue(True)
        os.remove(path)
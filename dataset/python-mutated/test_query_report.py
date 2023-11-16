import frappe
import frappe.utils
from frappe.core.doctype.doctype.test_doctype import new_doctype
from frappe.desk.query_report import build_xlsx_data, export_query, run
from frappe.tests.utils import FrappeTestCase
from frappe.utils.xlsxutils import make_xlsx

class TestQueryReport(FrappeTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        cls.enable_safe_exec()
        return super().setUpClass()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        frappe.db.rollback()

    def test_xlsx_data_with_multiple_datatypes(self):
        if False:
            print('Hello World!')
        'Test exporting report using rows with multiple datatypes (list, dict)'
        data = frappe._dict()
        data.columns = [{'label': 'Column A', 'fieldname': 'column_a', 'fieldtype': 'Float'}, {'label': 'Column B', 'fieldname': 'column_b', 'width': 100, 'fieldtype': 'Float'}, {'label': 'Column C', 'fieldname': 'column_c', 'width': 150, 'fieldtype': 'Duration'}]
        data.result = [[1.0, 3.0, 600], {'column_a': 22.1, 'column_b': 21.8, 'column_c': 86412}, {'column_b': 5.1, 'column_c': 53234, 'column_a': 11.1}, [3.0, 1.5, 333]]
        visible_idx = [0, 2, 3]
        (xlsx_data, column_widths) = build_xlsx_data(data, visible_idx, include_indentation=0)
        self.assertEqual(type(xlsx_data), list)
        self.assertEqual(len(xlsx_data), 4)
        self.assertListEqual(column_widths, [0, 10, 15])
        for row in xlsx_data:
            self.assertIsInstance(row, list)
        for row in xlsx_data[1:]:
            for cell in row:
                self.assertIsInstance(cell, (int, float))

    def test_xlsx_export_with_composite_cell_value(self):
        if False:
            for i in range(10):
                print('nop')
        'Test excel export using rows with composite cell value'
        data = frappe._dict()
        data.columns = [{'label': 'Column A', 'fieldname': 'column_a', 'fieldtype': 'Float'}, {'label': 'Column B', 'fieldname': 'column_b', 'width': 150, 'fieldtype': 'Data'}]
        data.result = [[1.0, 'Dummy 1'], {'column_a': 22.1, 'column_b': ['Dummy 1', 'Dummy 2']}]
        visible_idx = [0, 1]
        (xlsx_data, column_widths) = build_xlsx_data(data, visible_idx, include_indentation=0)
        make_xlsx(xlsx_data, 'Query Report', column_widths=column_widths)
        for row in xlsx_data:
            self.assertEqual(type(row[1]), str)

    def test_csv(self):
        if False:
            while True:
                i = 10
        from csv import QUOTE_ALL, QUOTE_MINIMAL, QUOTE_NONE, QUOTE_NONNUMERIC, DictReader
        from io import StringIO
        REPORT_NAME = 'Test CSV Report'
        REF_DOCTYPE = 'DocType'
        REPORT_COLUMNS = ['name', 'module', 'issingle']
        if not frappe.db.exists('Report', REPORT_NAME):
            report = frappe.new_doc('Report')
            report.report_name = REPORT_NAME
            report.ref_doctype = 'User'
            report.report_type = 'Query Report'
            report.query = frappe.qb.from_(REF_DOCTYPE).select(*REPORT_COLUMNS).limit(10).get_sql()
            report.is_standard = 'No'
            report.save()
        for delimiter in (',', ';', '\t', '|'):
            for quoting in (QUOTE_ALL, QUOTE_MINIMAL, QUOTE_NONE, QUOTE_NONNUMERIC):
                frappe.local.form_dict = frappe._dict({'report_name': REPORT_NAME, 'file_format_type': 'CSV', 'csv_quoting': quoting, 'csv_delimiter': delimiter, 'include_indentation': 0, 'visible_idx': [0, 1, 2]})
                export_query()
                self.assertTrue(frappe.response['filename'].endswith('.csv'))
                self.assertEqual(frappe.response['type'], 'binary')
                with StringIO(frappe.response['filecontent'].decode('utf-8')) as result:
                    reader = DictReader(result, delimiter=delimiter, quoting=quoting)
                    row = reader.__next__()
                    for column in REPORT_COLUMNS:
                        self.assertIn(column, row)
        frappe.delete_doc('Report', REPORT_NAME, delete_permanently=True)

    def test_report_for_duplicate_column_names(self):
        if False:
            while True:
                i = 10
        'Test report with duplicate column names'
        try:
            fields = [{'label': 'First Name', 'fieldname': 'first_name', 'fieldtype': 'Data'}, {'label': 'Last Name', 'fieldname': 'last_name', 'fieldtype': 'Data'}]
            docA = frappe.get_doc({'doctype': 'DocType', 'name': 'Doc A', 'module': 'Core', 'custom': 1, 'autoname': 'field:first_name', 'fields': fields, 'permissions': [{'role': 'System Manager'}]}).insert(ignore_if_duplicate=True)
            docB = frappe.get_doc({'doctype': 'DocType', 'name': 'Doc B', 'module': 'Core', 'custom': 1, 'autoname': 'field:last_name', 'fields': fields, 'permissions': [{'role': 'System Manager'}]}).insert(ignore_if_duplicate=True)
            for i in range(1, 3):
                frappe.get_doc({'doctype': 'Doc A', 'first_name': f'John{i}', 'last_name': 'Doe'}).insert()
                frappe.get_doc({'doctype': 'Doc B', 'last_name': f'Doe{i}', 'first_name': 'John'}).insert()
            if not frappe.db.exists('Report', 'Doc A Report'):
                report = frappe.get_doc({'doctype': 'Report', 'ref_doctype': 'Doc A', 'report_name': 'Doc A Report', 'report_type': 'Script Report', 'is_standard': 'No'}).insert(ignore_permissions=True)
            else:
                report = frappe.get_doc('Report', 'Doc A Report')
            report.report_script = '\nresult = [["Ritvik","Sardana", "Doe1"],["Shariq","Ansari", "Doe2"]]\ncolumns = [{\n\t\t\t"label": "First Name",\n\t\t\t"fieldname": "first_name",\n\t\t\t"fieldtype": "Data",\n\t\t\t"width": 180,\n\t\t},\n\t\t{\n\t\t\t"label": "Last Name",\n\t\t\t"fieldname": "last_name",\n\t\t\t"fieldtype": "Data",\n\t\t\t"width": 180,\n\t\t},\n\t\t{\n\t\t\t"label": "Linked Field",\n\t\t\t"fieldname": "linked_field",\n\t\t\t"fieldtype": "Link",\n\t\t\t"options": "Doc B",\n\t\t\t"width": 180,\n\t\t},\n\t]\n\ndata = columns, result\n\t\t\t\t'
            report.save()
            custom_columns = [{'fieldname': 'first_name-Doc_B', 'fieldtype': 'Data', 'label': 'First Name', 'insert_after_index': 1, 'link_field': {'fieldname': 'linked_field', 'names': {}}, 'doctype': 'Doc B', 'width': 100, 'id': 'first_name-Doc_B', 'name': 'First Name', 'editable': False, 'compareValue': None}]
            response = run('Doc A Report', filters={'user': 'Administrator', 'doctype': 'Doc A'}, custom_columns=custom_columns)
            self.assertListEqual(['first_name', 'last_name', 'first_name-Doc_B', 'linked_field'], [d['fieldname'] for d in response['columns']])
            self.assertDictEqual({'first_name': 'Ritvik', 'last_name': 'Sardana', 'linked_field': 'Doe1', 'first_name-Doc_B': 'John'}, response['result'][0])
        except Exception as e:
            raise e
            frappe.db.rollback()
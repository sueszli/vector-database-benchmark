import csv
import io
import unittest
from odoo.tests.common import TransactionCase, can_import
from odoo.modules.module import get_module_resource
from odoo.tools import mute_logger
ID_FIELD = {'id': 'id', 'name': 'id', 'string': 'External ID', 'required': False, 'fields': [], 'type': 'id'}

def make_field(name='value', string='Value', required=False, fields=[], field_type='id'):
    if False:
        print('Hello World!')
    return [ID_FIELD, {'id': name, 'name': name, 'string': string, 'required': required, 'fields': fields, 'type': field_type}]

def sorted_fields(fields):
    if False:
        print('Hello World!')
    ' recursively sort field lists to ease comparison '
    recursed = [dict(field, fields=sorted_fields(field['fields'])) for field in fields]
    return sorted(recursed, key=lambda field: field['id'])

class BaseImportCase(TransactionCase):

    def assertEqualFields(self, fields1, fields2):
        if False:
            return 10
        self.assertEqual(sorted_fields(fields1), sorted_fields(fields2))

class TestBasicFields(BaseImportCase):

    def get_fields(self, field):
        if False:
            print('Hello World!')
        return self.env['base_import.import'].get_fields('base_import.tests.models.' + field)

    def test_base(self):
        if False:
            while True:
                i = 10
        ' A basic field is not required '
        self.assertEqualFields(self.get_fields('char'), make_field(field_type='char'))

    def test_required(self):
        if False:
            for i in range(10):
                print('nop')
        ' Required fields should be flagged (so they can be fill-required) '
        self.assertEqualFields(self.get_fields('char.required'), make_field(required=True, field_type='char'))

    def test_readonly(self):
        if False:
            while True:
                i = 10
        ' Readonly fields should be filtered out'
        self.assertEqualFields(self.get_fields('char.readonly'), [ID_FIELD])

    def test_readonly_states(self):
        if False:
            for i in range(10):
                print('nop')
        ' Readonly fields with states should not be filtered out'
        self.assertEqualFields(self.get_fields('char.states'), make_field(field_type='char'))

    def test_readonly_states_noreadonly(self):
        if False:
            return 10
        ' Readonly fields with states having nothing to do with\n        readonly should still be filtered out'
        self.assertEqualFields(self.get_fields('char.noreadonly'), [ID_FIELD])

    def test_readonly_states_stillreadonly(self):
        if False:
            i = 10
            return i + 15
        ' Readonly fields with readonly states leaving them readonly\n        always... filtered out'
        self.assertEqualFields(self.get_fields('char.stillreadonly'), [ID_FIELD])

    def test_m2o(self):
        if False:
            return 10
        ' M2O fields should allow import of themselves (name_get),\n        their id and their xid'
        self.assertEqualFields(self.get_fields('m2o'), make_field(field_type='many2one', fields=[{'id': 'value', 'name': 'id', 'string': 'External ID', 'required': False, 'fields': [], 'type': 'id'}, {'id': 'value', 'name': '.id', 'string': 'Database ID', 'required': False, 'fields': [], 'type': 'id'}]))

    def test_m2o_required(self):
        if False:
            while True:
                i = 10
        ' If an m2o field is required, its three sub-fields are\n        required as well (the client has to handle that: requiredness\n        is id-based)\n        '
        self.assertEqualFields(self.get_fields('m2o.required'), make_field(field_type='many2one', required=True, fields=[{'id': 'value', 'name': 'id', 'string': 'External ID', 'required': True, 'fields': [], 'type': 'id'}, {'id': 'value', 'name': '.id', 'string': 'Database ID', 'required': True, 'fields': [], 'type': 'id'}]))

class TestO2M(BaseImportCase):

    def get_fields(self, field):
        if False:
            i = 10
            return i + 15
        return self.env['base_import.import'].get_fields('base_import.tests.models.' + field)

    def test_shallow(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqualFields(self.get_fields('o2m'), make_field(field_type='one2many', fields=[ID_FIELD, {'id': 'parent_id', 'name': 'parent_id', 'string': 'Parent id', 'type': 'many2one', 'required': False, 'fields': [{'id': 'parent_id', 'name': 'id', 'string': 'External ID', 'required': False, 'fields': [], 'type': 'id'}, {'id': 'parent_id', 'name': '.id', 'string': 'Database ID', 'required': False, 'fields': [], 'type': 'id'}]}, {'id': 'value', 'name': 'value', 'string': 'Value', 'required': False, 'fields': [], 'type': 'integer'}]))

class TestMatchHeadersSingle(TransactionCase):

    def test_match_by_name(self):
        if False:
            while True:
                i = 10
        match = self.env['base_import.import']._match_header('f0', [{'name': 'f0'}], {})
        self.assertEqual(match, [{'name': 'f0'}])

    def test_match_by_string(self):
        if False:
            i = 10
            return i + 15
        match = self.env['base_import.import']._match_header('some field', [{'name': 'bob', 'string': 'Some Field'}], {})
        self.assertEqual(match, [{'name': 'bob', 'string': 'Some Field'}])

    def test_nomatch(self):
        if False:
            for i in range(10):
                print('nop')
        match = self.env['base_import.import']._match_header('should not be', [{'name': 'bob', 'string': 'wheee'}], {})
        self.assertEqual(match, [])

    def test_recursive_match(self):
        if False:
            while True:
                i = 10
        f = {'name': 'f0', 'string': 'My Field', 'fields': [{'name': 'f0', 'string': 'Sub field 0', 'fields': []}, {'name': 'f1', 'string': 'Sub field 2', 'fields': []}]}
        match = self.env['base_import.import']._match_header('f0/f1', [f], {})
        self.assertEqual(match, [f, f['fields'][1]])

    def test_recursive_nomatch(self):
        if False:
            i = 10
            return i + 15
        ' Match first level, fail to match second level\n        '
        f = {'name': 'f0', 'string': 'My Field', 'fields': [{'name': 'f0', 'string': 'Sub field 0', 'fields': []}, {'name': 'f1', 'string': 'Sub field 2', 'fields': []}]}
        match = self.env['base_import.import']._match_header('f0/f2', [f], {})
        self.assertEqual(match, [])

class TestMatchHeadersMultiple(TransactionCase):

    def test_noheaders(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.env['base_import.import']._match_headers([], [], {}), (None, None))

    def test_nomatch(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.env['base_import.import']._match_headers(iter([['foo', 'bar', 'baz', 'qux'], ['v1', 'v2', 'v3', 'v4']]), [], {'headers': True}), (['foo', 'bar', 'baz', 'qux'], dict.fromkeys(range(4))))

    def test_mixed(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.env['base_import.import']._match_headers(iter(['foo bar baz qux/corge'.split()]), [{'name': 'bar', 'string': 'Bar'}, {'name': 'bob', 'string': 'Baz'}, {'name': 'qux', 'string': 'Qux', 'fields': [{'name': 'corge', 'fields': []}]}], {'headers': True}), (['foo', 'bar', 'baz', 'qux/corge'], {0: None, 1: ['bar'], 2: ['bob'], 3: ['qux', 'corge']}))

class TestPreview(TransactionCase):

    def make_import(self):
        if False:
            while True:
                i = 10
        import_wizard = self.env['base_import.import'].create({'res_model': 'res.users', 'file': u'로그인,언어\nbob,1\n'.encode('euc_kr'), 'file_type': 'text/csv', 'file_name': 'kr_data.csv'})
        return import_wizard

    @mute_logger('odoo.addons.base_import.models.base_import')
    def test_encoding(self):
        if False:
            return 10
        import_wizard = self.make_import()
        result = import_wizard.parse_preview({'quoting': '"', 'separator': ','})
        self.assertTrue('error' in result)

    @mute_logger('odoo.addons.base_import.models.base_import')
    def test_csv_errors(self):
        if False:
            print('Hello World!')
        import_wizard = self.make_import()
        result = import_wizard.parse_preview({'quoting': 'foo', 'separator': ',', 'encoding': 'euc_kr'})
        self.assertTrue('error' in result)
        result = import_wizard.parse_preview({'quoting': '"', 'separator': 'bob', 'encoding': 'euc_kr'})
        self.assertTrue('error' in result)

    def test_csv_success(self):
        if False:
            return 10
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': 'name,Some Value,Counter\nfoo,1,2\nbar,3,4\nqux,5,6\n', 'file_type': 'text/csv'})
        result = import_wizard.parse_preview({'quoting': '"', 'separator': ',', 'headers': True})
        self.assertIsNone(result.get('error'))
        self.assertEqual(result['matches'], {0: ['name'], 1: ['somevalue'], 2: None})
        self.assertEqual(result['headers'], ['name', 'Some Value', 'Counter'])
        self.assertItemsEqual(result['fields'], [ID_FIELD, {'id': 'name', 'name': 'name', 'string': 'Name', 'required': False, 'fields': [], 'type': 'char'}, {'id': 'somevalue', 'name': 'somevalue', 'string': 'Some Value', 'required': True, 'fields': [], 'type': 'integer'}, {'id': 'othervalue', 'name': 'othervalue', 'string': 'Other Variable', 'required': False, 'fields': [], 'type': 'integer'}])
        self.assertEqual(result['preview'], [['foo', '1', '2'], ['bar', '3', '4'], ['qux', '5', '6']])
        self.assertItemsEqual(result.keys(), ['matches', 'headers', 'fields', 'preview', 'headers_type', 'options', 'advanced_mode', 'debug'])

    @unittest.skipUnless(can_import('xlrd'), 'XLRD module not available')
    def test_xls_success(self):
        if False:
            for i in range(10):
                print('nop')
        xls_file_path = get_module_resource('base_import', 'tests', 'test.xls')
        file_content = open(xls_file_path, 'rb').read()
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': file_content, 'file_type': 'application/vnd.ms-excel'})
        result = import_wizard.parse_preview({'headers': True})
        self.assertIsNone(result.get('error'))
        self.assertEqual(result['matches'], {0: ['name'], 1: ['somevalue'], 2: None})
        self.assertEqual(result['headers'], ['name', 'Some Value', 'Counter'])
        self.assertItemsEqual(result['fields'], [ID_FIELD, {'id': 'name', 'name': 'name', 'string': 'Name', 'required': False, 'fields': [], 'type': 'char'}, {'id': 'somevalue', 'name': 'somevalue', 'string': 'Some Value', 'required': True, 'fields': [], 'type': 'integer'}, {'id': 'othervalue', 'name': 'othervalue', 'string': 'Other Variable', 'required': False, 'fields': [], 'type': 'integer'}])
        self.assertEqual(result['preview'], [['foo', '1', '2'], ['bar', '3', '4'], ['qux', '5', '6']])
        self.assertItemsEqual(result.keys(), ['matches', 'headers', 'fields', 'preview', 'headers_type', 'options', 'advanced_mode', 'debug'])

    @unittest.skipUnless(can_import('xlrd.xlsx'), 'XLRD/XLSX not available')
    def test_xlsx_success(self):
        if False:
            i = 10
            return i + 15
        xlsx_file_path = get_module_resource('base_import', 'tests', 'test.xlsx')
        file_content = open(xlsx_file_path, 'rb').read()
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': file_content, 'file_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'})
        result = import_wizard.parse_preview({'headers': True})
        self.assertIsNone(result.get('error'))
        self.assertEqual(result['matches'], {0: ['name'], 1: ['somevalue'], 2: None})
        self.assertEqual(result['headers'], ['name', 'Some Value', 'Counter'])
        self.assertItemsEqual(result['fields'], [ID_FIELD, {'id': 'name', 'name': 'name', 'string': 'Name', 'required': False, 'fields': [], 'type': 'char'}, {'id': 'somevalue', 'name': 'somevalue', 'string': 'Some Value', 'required': True, 'fields': [], 'type': 'integer'}, {'id': 'othervalue', 'name': 'othervalue', 'string': 'Other Variable', 'required': False, 'fields': [], 'type': 'integer'}])
        self.assertEqual(result['preview'], [['foo', '1', '2'], ['bar', '3', '4'], ['qux', '5', '6']])
        self.assertItemsEqual(result.keys(), ['matches', 'headers', 'fields', 'preview', 'headers_type', 'options', 'advanced_mode', 'debug'])

    @unittest.skipUnless(can_import('odf'), 'ODFPY not available')
    def test_ods_success(self):
        if False:
            return 10
        ods_file_path = get_module_resource('base_import', 'tests', 'test.ods')
        file_content = open(ods_file_path, 'rb').read()
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': file_content, 'file_type': 'application/vnd.oasis.opendocument.spreadsheet'})
        result = import_wizard.parse_preview({'headers': True})
        self.assertIsNone(result.get('error'))
        self.assertEqual(result['matches'], {0: ['name'], 1: ['somevalue'], 2: None})
        self.assertEqual(result['headers'], ['name', 'Some Value', 'Counter'])
        self.assertItemsEqual(result['fields'], [ID_FIELD, {'id': 'name', 'name': 'name', 'string': 'Name', 'required': False, 'fields': [], 'type': 'char'}, {'id': 'somevalue', 'name': 'somevalue', 'string': 'Some Value', 'required': True, 'fields': [], 'type': 'integer'}, {'id': 'othervalue', 'name': 'othervalue', 'string': 'Other Variable', 'required': False, 'fields': [], 'type': 'integer'}])
        self.assertEqual(result['preview'], [['foo', '1', '2'], ['bar', '3', '4'], ['aux', '5', '6']])
        self.assertItemsEqual(result.keys(), ['matches', 'headers', 'fields', 'preview', 'headers_type', 'options', 'advanced_mode', 'debug'])

class test_convert_import_data(TransactionCase):
    """ Tests conversion of base_import.import input into data which
    can be fed to Model.load
    """

    def test_all(self):
        if False:
            print('Hello World!')
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': 'name,Some Value,Counter\nfoo,1,2\nbar,3,4\nqux,5,6\n', 'file_type': 'text/csv'})
        (data, fields) = import_wizard._convert_import_data(['name', 'somevalue', 'othervalue'], {'quoting': '"', 'separator': ',', 'headers': True})
        self.assertItemsEqual(fields, ['name', 'somevalue', 'othervalue'])
        self.assertItemsEqual(data, [['foo', '1', '2'], ['bar', '3', '4'], ['qux', '5', '6']])

    def test_date_fields(self):
        if False:
            for i in range(10):
                print('nop')
        import_wizard = self.env['base_import.import'].create({'res_model': 'res.partner', 'file': 'name,date,create_date\n"foo","2013年07月18日","2016-10-12 06:06"\n', 'file_type': 'text/csv'})
        results = import_wizard.do(['name', 'date', 'create_date'], {'date_format': '%Y年%m月%d日', 'datetime_format': '%Y-%m-%d %H:%M', 'quoting': '"', 'separator': ',', 'headers': True})
        self.assertItemsEqual(results, [])

    def test_filtered(self):
        if False:
            for i in range(10):
                print('nop')
        ' If ``False`` is provided as field mapping for a column,\n        that column should be removed from importable data\n        '
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': 'name,Some Value,Counter\nfoo,1,2\nbar,3,4\nqux,5,6\n', 'file_type': 'text/csv'})
        (data, fields) = import_wizard._convert_import_data(['name', False, 'othervalue'], {'quoting': '"', 'separator': ',', 'headers': True})
        self.assertItemsEqual(fields, ['name', 'othervalue'])
        self.assertItemsEqual(data, [['foo', '2'], ['bar', '4'], ['qux', '6']])

    def test_norow(self):
        if False:
            i = 10
            return i + 15
        ' If a row is composed only of empty values (due to having\n        filtered out non-empty values from it), it should be removed\n        '
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': 'name,Some Value,Counter\nfoo,1,2\n,3,\n,5,6\n', 'file_type': 'text/csv'})
        (data, fields) = import_wizard._convert_import_data(['name', False, 'othervalue'], {'quoting': '"', 'separator': ',', 'headers': True})
        self.assertItemsEqual(fields, ['name', 'othervalue'])
        self.assertItemsEqual(data, [['foo', '2'], ['', '6']])

    def test_empty_rows(self):
        if False:
            while True:
                i = 10
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': 'name,Some Value\nfoo,1\n\nbar,2\n     \n\t \n', 'file_type': 'text/csv'})
        (data, fields) = import_wizard._convert_import_data(['name', 'somevalue'], {'quoting': '"', 'separator': ',', 'headers': True})
        self.assertItemsEqual(fields, ['name', 'somevalue'])
        self.assertItemsEqual(data, [['foo', '1'], ['bar', '2']])

    def test_nofield(self):
        if False:
            while True:
                i = 10
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': 'name,Some Value,Counter\nfoo,1,2\n', 'file_type': 'text/csv'})
        self.assertRaises(ValueError, import_wizard._convert_import_data, [], {'quoting': '"', 'separator': ',', 'headers': True})

    def test_falsefields(self):
        if False:
            return 10
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': 'name,Some Value,Counter\nfoo,1,2\n', 'file_type': 'text/csv'})
        self.assertRaises(ValueError, import_wizard._convert_import_data, [False, False, False], {'quoting': '"', 'separator': ',', 'headers': True})

    def test_newline_import(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure importing keep newlines\n        '
        output = io.BytesIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        data_row = ['\tfoo\n\tbar', ' "hello" \n\n \'world\' ']
        writer.writerow(['name', 'Some Value'])
        writer.writerow(data_row)
        import_wizard = self.env['base_import.import'].create({'res_model': 'base_import.tests.models.preview', 'file': output.getvalue(), 'file_type': 'text/csv'})
        (data, _) = import_wizard._convert_import_data(['name', 'somevalue'], {'quoting': '"', 'separator': ',', 'headers': True})
        self.assertItemsEqual(data, [data_row])

class test_failures(TransactionCase):

    def test_big_attachments(self):
        if False:
            while True:
                i = 10
        "\n        Ensure big fields (e.g. b64-encoded image data) can be imported and\n        we're not hitting limits of the default CSV parser config\n        "
        import csv, cStringIO
        from PIL import Image
        im = Image.new('RGB', (1920, 1080))
        fout = cStringIO.StringIO()
        writer = csv.writer(fout, dialect=None)
        writer.writerows([['name', 'db_datas'], ['foo', im.tobytes().encode('base64')]])
        import_wizard = self.env['base_import.import'].create({'res_model': 'ir.attachment', 'file': fout.getvalue(), 'file_type': 'text/csv'})
        results = import_wizard.do(['name', 'db_datas'], {'headers': True, 'separator': ',', 'quoting': '"'})
        self.assertFalse(results, 'results should be empty on successful import')
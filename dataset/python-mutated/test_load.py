import json
import pkgutil
from odoo.tests import common
from odoo.tools.misc import mute_logger

def message(msg, type='error', from_=0, to_=0, record=0, field='value', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return dict(kwargs, type=type, rows={'from': from_, 'to': to_}, record=record, field=field, message=msg)

def moreaction(**kwargs):
    if False:
        return 10
    return dict(kwargs, type='ir.actions.act_window', target='new', view_mode='tree,form', view_type='form', views=[(False, 'tree'), (False, 'form')], help=u'See all possible values')

def values(seq, field='value'):
    if False:
        i = 10
        return i + 15
    return [item[field] for item in seq]

class ImporterCase(common.TransactionCase):
    model_name = False

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(ImporterCase, self).__init__(*args, **kwargs)
        self.model = None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(ImporterCase, self).setUp()
        self.model = self.env[self.model_name]
        self.env['ir.model.data'].clear_caches()

    def import_(self, fields, rows, context=None):
        if False:
            i = 10
            return i + 15
        return self.model.with_context(context or {}).load(fields, rows)

    def read(self, fields=('value',), domain=(), context=None):
        if False:
            while True:
                i = 10
        records = self.model.with_context(context or {}).search(domain)
        return records.read(fields)

    def browse(self, domain=(), context=None):
        if False:
            for i in range(10):
                print('nop')
        return self.model.with_context(context or {}).search(domain)

    def xid(self, record):
        if False:
            print('Hello World!')
        ModelData = self.env['ir.model.data']
        data = ModelData.search([('model', '=', record._name), ('res_id', '=', record.id)])
        if data:
            d = data.read(['name', 'module'])[0]
            if d['module']:
                return '%s.%s' % (d['module'], d['name'])
            return d['name']
        name = record.name_get()[0][1]
        name = name.replace('.', '-')
        ModelData.create({'name': name, 'model': record._name, 'res_id': record.id, 'module': '__test__'})
        return '__test__.' + name

    def add_translations(self, name, type, code, *tnx):
        if False:
            i = 10
            return i + 15
        self.env['res.lang'].load_lang(code)
        Translations = self.env['ir.translation']
        for (source, value) in tnx:
            Translations.create({'name': name, 'lang': code, 'type': type, 'src': source, 'value': value, 'state': 'translated'})

class test_ids_stuff(ImporterCase):
    model_name = 'export.integer'

    def test_create_with_id(self):
        if False:
            i = 10
            return i + 15
        result = self.import_(['.id', 'value'], [['42', '36']])
        self.assertIs(result['ids'], False)
        self.assertEqual(result['messages'], [{'type': 'error', 'rows': {'from': 0, 'to': 0}, 'record': 0, 'field': '.id', 'message': u"Unknown database identifier '42'"}])

    def test_create_with_xid(self):
        if False:
            i = 10
            return i + 15
        result = self.import_(['id', 'value'], [['somexmlid', '42']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        self.assertEqual('somexmlid', self.xid(self.browse()[0]))

    def test_update_with_id(self):
        if False:
            i = 10
            return i + 15
        record = self.model.create({'value': 36})
        self.assertEqual(36, record.value)
        result = self.import_(['.id', 'value'], [[str(record.id), '42']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        self.assertEqual([42], values(self.read()))

    def test_update_with_xid(self):
        if False:
            return 10
        self.import_(['id', 'value'], [['somexmlid', '36']])
        self.assertEqual([36], values(self.read()))
        self.import_(['id', 'value'], [['somexmlid', '1234567']])
        self.assertEqual([1234567], values(self.read()))

class test_boolean_field(ImporterCase):
    model_name = 'export.boolean'

    def test_empty(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.import_(['value'], []), {'ids': [], 'messages': []})

    def test_exported(self):
        if False:
            print('Hello World!')
        result = self.import_(['value'], [['False'], ['True']])
        self.assertEqual(len(result['ids']), 2)
        self.assertFalse(result['messages'])
        records = self.read()
        self.assertEqual([False, True], values(records))

    def test_falses(self):
        if False:
            i = 10
            return i + 15
        for (lang, source, value) in [('fr_FR', 'no', u'non'), ('de_DE', 'no', u'nein'), ('ru_RU', 'no', u'нет'), ('nl_BE', 'false', u'vals'), ('lt_LT', 'false', u'klaidingas')]:
            self.add_translations('test_import.py', 'code', lang, (source, value))
        falses = [[u'0'], [u'no'], [u'false'], [u'FALSE'], [u''], [u'non'], [u'nein'], [u'нет'], [u'vals'], [u'klaidingas']]
        result = self.import_(['value'], falses)
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), len(falses))
        self.assertEqual([False] * len(falses), values(self.read()))

    def test_trues(self):
        if False:
            print('Hello World!')
        trues = [['None'], ['nil'], ['()'], ['f'], ['#f'], ['VRAI'], ['ok'], ['true'], ['yes'], ['1']]
        result = self.import_(['value'], trues)
        self.assertEqual(len(result['ids']), 10)
        self.assertEqual(result['messages'], [message(u"Unknown value '%s' for boolean field 'Value', assuming 'yes'" % v[0], moreinfo=u"Use '1' for yes and '0' for no", type='warning', from_=i, to_=i, record=i) for (i, v) in enumerate(trues) if v[0] not in ('true', 'yes', '1')])
        self.assertEqual([True] * 10, values(self.read()))

class test_integer_field(ImporterCase):
    model_name = 'export.integer'

    def test_none(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.import_(['value'], []), {'ids': [], 'messages': []})

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.import_(['value'], [['']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        self.assertEqual([False], values(self.read()))

    def test_zero(self):
        if False:
            i = 10
            return i + 15
        result = self.import_(['value'], [['0']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        result = self.import_(['value'], [['-0']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        self.assertEqual([False, False], values(self.read()))

    def test_positives(self):
        if False:
            print('Hello World!')
        result = self.import_(['value'], [['1'], ['42'], [str(2 ** 31 - 1)], ['12345678']])
        self.assertEqual(len(result['ids']), 4)
        self.assertFalse(result['messages'])
        self.assertEqual([1, 42, 2 ** 31 - 1, 12345678], values(self.read()))

    def test_negatives(self):
        if False:
            return 10
        result = self.import_(['value'], [['-1'], ['-42'], [str(-(2 ** 31 - 1))], [str(-2 ** 31)], ['-12345678']])
        self.assertEqual(len(result['ids']), 5)
        self.assertFalse(result['messages'])
        self.assertEqual([-1, -42, -(2 ** 31 - 1), -2 ** 31, -12345678], values(self.read()))

    @mute_logger('odoo.sql_db', 'odoo.models')
    def test_out_of_range(self):
        if False:
            print('Hello World!')
        result = self.import_(['value'], [[str(2 ** 31)]])
        self.assertIs(result['ids'], False)
        self.assertEqual(result['messages'], [{'type': 'error', 'rows': {'from': 0, 'to': 0}, 'record': 0, 'message': 'integer out of range\n'}])
        result = self.import_(['value'], [[str(-2 ** 32)]])
        self.assertIs(result['ids'], False)
        self.assertEqual(result['messages'], [{'type': 'error', 'rows': {'from': 0, 'to': 0}, 'record': 0, 'message': 'integer out of range\n'}])

    def test_nonsense(self):
        if False:
            return 10
        result = self.import_(['value'], [['zorglub']])
        self.assertIs(result['ids'], False)
        self.assertEqual(result['messages'], [{'type': 'error', 'rows': {'from': 0, 'to': 0}, 'record': 0, 'field': 'value', 'message': u"'zorglub' does not seem to be an integer for field 'Value'"}])

class test_float_field(ImporterCase):
    model_name = 'export.float'

    def test_none(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.import_(['value'], []), {'ids': [], 'messages': []})

    def test_empty(self):
        if False:
            return 10
        result = self.import_(['value'], [['']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        self.assertEqual([False], values(self.read()))

    def test_zero(self):
        if False:
            i = 10
            return i + 15
        result = self.import_(['value'], [['0']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        result = self.import_(['value'], [['-0']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        self.assertEqual([False, False], values(self.read()))

    def test_positives(self):
        if False:
            return 10
        result = self.import_(['value'], [['1'], ['42'], [str(2 ** 31 - 1)], ['12345678'], [str(2 ** 33)], ['0.000001']])
        self.assertEqual(len(result['ids']), 6)
        self.assertFalse(result['messages'])
        self.assertEqual([1, 42, 2 ** 31 - 1, 12345678, 2.0 ** 33, 1e-06], values(self.read()))

    def test_negatives(self):
        if False:
            i = 10
            return i + 15
        result = self.import_(['value'], [['-1'], ['-42'], [str(-2 ** 31 + 1)], [str(-2 ** 31)], ['-12345678'], [str(-2 ** 33)], ['-0.000001']])
        self.assertEqual(len(result['ids']), 7)
        self.assertFalse(result['messages'])
        self.assertEqual([-1, -42, -(2 ** 31 - 1), -2 ** 31, -12345678, -2.0 ** 33, -1e-06], values(self.read()))

    def test_nonsense(self):
        if False:
            while True:
                i = 10
        result = self.import_(['value'], [['foobar']])
        self.assertIs(result['ids'], False)
        self.assertEqual(result['messages'], [message(u"'foobar' does not seem to be a number for field 'Value'")])

class test_string_field(ImporterCase):
    model_name = 'export.string.bounded'

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.import_(['value'], [['']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        self.assertEqual([False], values(self.read()))

    def test_imported(self):
        if False:
            while True:
                i = 10
        result = self.import_(['value'], [[u'foobar'], [u'foobarbaz'], [u'Með suð í eyrum við spilum endalaust'], [u"People 'get' types. They use them all the time. Telling someone he can't pound a nail with a banana doesn't much surprise him."]])
        self.assertEqual(len(result['ids']), 4)
        self.assertFalse(result['messages'])
        self.assertEqual([u'foobar', u'foobarbaz', u'Með suð í eyrum ', u"People 'get' typ"], values(self.read()))

class test_unbound_string_field(ImporterCase):
    model_name = 'export.string'

    def test_imported(self):
        if False:
            i = 10
            return i + 15
        result = self.import_(['value'], [[u'í dag viðrar vel til loftárása'], [u'If they ask you about fun, you tell them – fun is a filthy parasite']])
        self.assertEqual(len(result['ids']), 2)
        self.assertFalse(result['messages'])
        self.assertEqual([u'í dag viðrar vel til loftárása', u'If they ask you about fun, you tell them – fun is a filthy parasite'], values(self.read()))

class test_required_string_field(ImporterCase):
    model_name = 'export.string.required'

    @mute_logger('odoo.sql_db', 'odoo.models')
    def test_empty(self):
        if False:
            print('Hello World!')
        result = self.import_(['value'], [[]])
        self.assertEqual(result['messages'], [message(u"Missing required value for the field 'Value' (value)")])
        self.assertIs(result['ids'], False)

    @mute_logger('odoo.sql_db', 'odoo.models')
    def test_not_provided(self):
        if False:
            print('Hello World!')
        result = self.import_(['const'], [['12']])
        self.assertEqual(result['messages'], [message(u"Missing required value for the field 'Value' (value)")])
        self.assertIs(result['ids'], False)

class test_text(ImporterCase):
    model_name = 'export.text'

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.import_(['value'], [['']])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        self.assertEqual([False], values(self.read()))

    def test_imported(self):
        if False:
            while True:
                i = 10
        s = u'Breiðskífa er notað um útgefna hljómplötu sem inniheldur stúdíóupptökur frá einum flytjanda. Breiðskífur eru oftast milli 25-80 mínútur og er lengd þeirra oft miðuð við 33⅓ snúninga 12 tommu vínylplötur (sem geta verið allt að 30 mín hvor hlið).\n\nBreiðskífur eru stundum tvöfaldar og eru þær þá gefnar út á tveimur geisladiskum eða tveimur vínylplötum.'
        result = self.import_(['value'], [[s]])
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])
        self.assertEqual([s], values(self.read()))

class test_selection(ImporterCase):
    model_name = 'export.selection'
    translations_fr = [('Foo', 'tete'), ('Bar', 'titi'), ('Qux', 'toto')]

    def test_imported(self):
        if False:
            i = 10
            return i + 15
        result = self.import_(['value'], [['Qux'], ['Bar'], ['Foo'], ['2']])
        self.assertEqual(len(result['ids']), 4)
        self.assertFalse(result['messages'])
        self.assertEqual([3, 2, 1, 2], values(self.read()))

    def test_imported_translated(self):
        if False:
            while True:
                i = 10
        self.add_translations('export.selection,value', 'selection', 'fr_FR', *self.translations_fr)
        result = self.import_(['value'], [['toto'], ['tete'], ['titi']], context={'lang': 'fr_FR'})
        self.assertEqual(len(result['ids']), 3)
        self.assertFalse(result['messages'])
        self.assertEqual([3, 1, 2], values(self.read()))
        result = self.import_(['value'], [['Foo']], context={'lang': 'fr_FR'})
        self.assertEqual(len(result['ids']), 1)
        self.assertFalse(result['messages'])

    def test_invalid(self):
        if False:
            return 10
        result = self.import_(['value'], [['Baz']])
        self.assertIs(result['ids'], False)
        self.assertEqual(result['messages'], [message(u"Value 'Baz' not found in selection field 'Value'", moreinfo='Foo Bar Qux 4'.split())])
        result = self.import_(['value'], [[42]])
        self.assertIs(result['ids'], False)
        self.assertEqual(result['messages'], [message(u"Value '42' not found in selection field 'Value'", moreinfo='Foo Bar Qux 4'.split())])

class test_selection_with_default(ImporterCase):
    model_name = 'export.selection.withdefault'

    def test_empty(self):
        if False:
            print('Hello World!')
        ' Empty cells should set corresponding field to False\n        '
        result = self.import_(['value'], [['']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        self.assertEqual(values(self.read()), [False])

    def test_default(self):
        if False:
            i = 10
            return i + 15
        ' Non-provided cells should set corresponding field to default\n        '
        result = self.import_(['const'], [['42']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        self.assertEqual(values(self.read()), [2])

class test_selection_function(ImporterCase):
    model_name = 'export.selection.function'
    translations_fr = [('Corge', 'toto'), ('Grault', 'titi'), ('Wheee', 'tete'), ('Moog', 'tutu')]

    def test_imported(self):
        if False:
            i = 10
            return i + 15
        ' import uses fields_get, so translates import label (may or may not\n        be good news) *and* serializes the selection function to reverse it:\n        import does not actually know that the selection field uses a function\n        '
        result = self.import_(['value'], [['3'], ['Grault']])
        self.assertEqual(len(result['ids']), 2)
        self.assertFalse(result['messages'])
        self.assertEqual(values(self.read()), ['3', '1'])

    def test_translated(self):
        if False:
            i = 10
            return i + 15
        ' Expects output of selection function returns translated labels\n        '
        self.add_translations('export.selection,value', 'selection', 'fr_FR', *self.translations_fr)
        result = self.import_(['value'], [['titi'], ['tete']], context={'lang': 'fr_FR'})
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 2)
        self.assertEqual(values(self.read()), ['1', '2'])
        result = self.import_(['value'], [['Wheee']], context={'lang': 'fr_FR'})
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)

class test_m2o(ImporterCase):
    model_name = 'export.many2one'

    def test_by_name(self):
        if False:
            return 10
        record1 = self.env['export.integer'].create({'value': 42})
        record2 = self.env['export.integer'].create({'value': 36})
        name1 = dict(record1.name_get())[record1.id]
        name2 = dict(record2.name_get())[record2.id]
        result = self.import_(['value'], [[name1], [name1], [name2]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 3)
        self.assertEqual([(record1.id, name1), (record1.id, name1), (record2.id, name2)], values(self.read()))

    def test_by_xid(self):
        if False:
            i = 10
            return i + 15
        record = self.env['export.integer'].create({'value': 42})
        xid = self.xid(record)
        result = self.import_(['value/id'], [[xid]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        b = self.browse()
        self.assertEqual(42, b[0].value.value)

    def test_by_id(self):
        if False:
            i = 10
            return i + 15
        record = self.env['export.integer'].create({'value': 42})
        result = self.import_(['value/.id'], [[record.id]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        b = self.browse()
        self.assertEqual(42, b[0].value.value)

    def test_by_names(self):
        if False:
            print('Hello World!')
        record1 = self.env['export.integer'].create({'value': 42})
        record2 = self.env['export.integer'].create({'value': 42})
        name1 = dict(record1.name_get())[record1.id]
        name2 = dict(record2.name_get())[record2.id]
        self.assertEqual(name1, name2)
        result = self.import_(['value'], [[name2]])
        self.assertEqual(result['messages'], [message(u"Found multiple matches for field 'Value' (2 matches)", type='warning')])
        self.assertEqual(len(result['ids']), 1)
        self.assertEqual([(record1.id, name1)], values(self.read()))

    def test_fail_by_implicit_id(self):
        if False:
            i = 10
            return i + 15
        " Can't implicitly import records by id\n        "
        record1 = self.env['export.integer'].create({'value': 42})
        record2 = self.env['export.integer'].create({'value': 36})
        result = self.import_(['value'], [[record1.id], [record2.id], [record1.id]])
        self.assertEqual(result['messages'], [message(u"No matching record found for name '%s' in field 'Value'" % id, from_=index, to_=index, record=index, moreinfo=moreaction(res_model='export.integer')) for (index, id) in enumerate([record1.id, record2.id, record1.id])])
        self.assertIs(result['ids'], False)

    @mute_logger('odoo.sql_db')
    def test_fail_id_mistype(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.import_(['value/.id'], [['foo']])
        self.assertEqual(result['messages'], [message(u"Invalid database id 'foo' for the field 'Value'", moreinfo=moreaction(res_model='ir.model.data', domain=[('model', '=', 'export.integer')]))])
        self.assertIs(result['ids'], False)

    def test_sub_field(self):
        if False:
            while True:
                i = 10
        " Does not implicitly create the record, does not warn that you can't\n        import m2o subfields (at all)...\n        "
        result = self.import_(['value/value'], [['42']])
        self.assertEqual(result['messages'], [message(u'Can not create Many-To-One records indirectly, import the field separately')])
        self.assertIs(result['ids'], False)

    def test_fail_noids(self):
        if False:
            return 10
        result = self.import_(['value'], [['nameisnoexist:3']])
        self.assertEqual(result['messages'], [message(u"No matching record found for name 'nameisnoexist:3' in field 'Value'", moreinfo=moreaction(res_model='export.integer'))])
        self.assertIs(result['ids'], False)
        result = self.import_(['value/id'], [['noxidhere']])
        self.assertEqual(result['messages'], [message(u"No matching record found for external id 'noxidhere' in field 'Value'", moreinfo=moreaction(res_model='ir.model.data', domain=[('model', '=', 'export.integer')]))])
        self.assertIs(result['ids'], False)
        result = self.import_(['value/.id'], [['66']])
        self.assertEqual(result['messages'], [message(u"No matching record found for database id '66' in field 'Value'", moreinfo=moreaction(res_model='ir.model.data', domain=[('model', '=', 'export.integer')]))])
        self.assertIs(result['ids'], False)

    def test_fail_multiple(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.import_(['value', 'value/id'], [['somename', 'somexid']])
        self.assertEqual(result['messages'], [message(u"Ambiguous specification for field 'Value', only provide one of name, external id or database id")])
        self.assertIs(result['ids'], False)

class test_m2m(ImporterCase):
    model_name = 'export.many2many'

    def test_ids(self):
        if False:
            while True:
                i = 10
        id1 = self.env['export.many2many.other'].create({'value': 3, 'str': 'record0'}).id
        id2 = self.env['export.many2many.other'].create({'value': 44, 'str': 'record1'}).id
        id3 = self.env['export.many2many.other'].create({'value': 84, 'str': 'record2'}).id
        id4 = self.env['export.many2many.other'].create({'value': 9, 'str': 'record3'}).id
        id5 = self.env['export.many2many.other'].create({'value': 99, 'str': 'record4'}).id
        result = self.import_(['value/.id'], [['%d,%d' % (id1, id2)], ['%d,%d,%d' % (id1, id3, id4)], ['%d,%d,%d' % (id1, id2, id3)], ['%d' % id5]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 4)
        ids = lambda records: [record.id for record in records]
        b = self.browse()
        self.assertEqual(ids(b[0].value), [id1, id2])
        self.assertEqual(values(b[0].value), [3, 44])
        self.assertEqual(ids(b[2].value), [id1, id2, id3])
        self.assertEqual(values(b[2].value), [3, 44, 84])

    def test_noids(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.import_(['value/.id'], [['42']])
        self.assertEqual(result['messages'], [message(u"No matching record found for database id '42' in field 'Value'", moreinfo=moreaction(res_model='ir.model.data', domain=[('model', '=', 'export.many2many.other')]))])
        self.assertIs(result['ids'], False)

    def test_xids(self):
        if False:
            print('Hello World!')
        record0 = self.env['export.many2many.other'].create({'value': 3, 'str': 'record0'})
        record1 = self.env['export.many2many.other'].create({'value': 44, 'str': 'record1'})
        record2 = self.env['export.many2many.other'].create({'value': 84, 'str': 'record2'})
        record3 = self.env['export.many2many.other'].create({'value': 9, 'str': 'record3'})
        result = self.import_(['value/id'], [['%s,%s' % (self.xid(record0), self.xid(record1))], ['%s' % self.xid(record3)], ['%s,%s' % (self.xid(record2), self.xid(record1))]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 3)
        b = self.browse()
        self.assertEqual(values(b[0].value), [3, 44])
        self.assertEqual(values(b[2].value), [44, 84])

    def test_noxids(self):
        if False:
            print('Hello World!')
        result = self.import_(['value/id'], [['noxidforthat']])
        self.assertEqual(result['messages'], [message(u"No matching record found for external id 'noxidforthat' in field 'Value'", moreinfo=moreaction(res_model='ir.model.data', domain=[('model', '=', 'export.many2many.other')]))])
        self.assertIs(result['ids'], False)

    def test_names(self):
        if False:
            return 10
        record0 = self.env['export.many2many.other'].create({'value': 3, 'str': 'record0'})
        record1 = self.env['export.many2many.other'].create({'value': 44, 'str': 'record1'})
        record2 = self.env['export.many2many.other'].create({'value': 84, 'str': 'record2'})
        record3 = self.env['export.many2many.other'].create({'value': 9, 'str': 'record3'})
        name = lambda record: record.name_get()[0][1]
        result = self.import_(['value'], [['%s,%s' % (name(record1), name(record2))], ['%s,%s,%s' % (name(record0), name(record1), name(record2))], ['%s,%s' % (name(record0), name(record3))]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 3)
        b = self.browse()
        self.assertEqual(values(b[1].value), [3, 44, 84])
        self.assertEqual(values(b[2].value), [3, 9])

    def test_nonames(self):
        if False:
            while True:
                i = 10
        result = self.import_(['value'], [['wherethem2mhavenonames']])
        self.assertEqual(result['messages'], [message(u"No matching record found for name 'wherethem2mhavenonames' in field 'Value'", moreinfo=moreaction(res_model='export.many2many.other'))])
        self.assertIs(result['ids'], False)

    def test_import_to_existing(self):
        if False:
            i = 10
            return i + 15
        id1 = self.env['export.many2many.other'].create({'value': 3, 'str': 'record0'}).id
        id2 = self.env['export.many2many.other'].create({'value': 44, 'str': 'record1'}).id
        id3 = self.env['export.many2many.other'].create({'value': 84, 'str': 'record2'}).id
        id4 = self.env['export.many2many.other'].create({'value': 9, 'str': 'record3'}).id
        xid = 'myxid'
        result = self.import_(['id', 'value/.id'], [[xid, '%d,%d' % (id1, id2)]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        result = self.import_(['id', 'value/.id'], [[xid, '%d,%d' % (id3, id4)]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        b = self.browse()
        self.assertEqual(len(b), 1)
        self.assertEqual(values(b[0].value), [84, 9])

class test_o2m(ImporterCase):
    model_name = 'export.one2many'

    def test_name_get(self):
        if False:
            for i in range(10):
                print('nop')
        s = u'Java is a DSL for taking large XML files and converting them to stack traces'
        result = self.import_(['const', 'value'], [['5', s]])
        self.assertEqual(result['messages'], [message(u"No matching record found for name '%s' in field 'Value'" % s, moreinfo=moreaction(res_model='export.one2many.child'))])
        self.assertIs(result['ids'], False)

    def test_single(self):
        if False:
            return 10
        result = self.import_(['const', 'value/value'], [['5', '63']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        (b,) = self.browse()
        self.assertEqual(b.const, 5)
        self.assertEqual(values(b.value), [63])

    def test_multicore(self):
        if False:
            while True:
                i = 10
        result = self.import_(['const', 'value/value'], [['5', '63'], ['6', '64']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 2)
        (b1, b2) = self.browse()
        self.assertEqual(b1.const, 5)
        self.assertEqual(values(b1.value), [63])
        self.assertEqual(b2.const, 6)
        self.assertEqual(values(b2.value), [64])

    def test_multisub(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.import_(['const', 'value/value'], [['5', '63'], ['', '64'], ['', '65'], ['', '66']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        (b,) = self.browse()
        self.assertEqual(values(b.value), [63, 64, 65, 66])

    def test_multi_subfields(self):
        if False:
            i = 10
            return i + 15
        result = self.import_(['value/str', 'const', 'value/value'], [['this', '5', '63'], ['is', '', '64'], ['the', '', '65'], ['rhythm', '', '66']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        (b,) = self.browse()
        self.assertEqual(values(b.value), [63, 64, 65, 66])
        self.assertEqual(values(b.value, 'str'), 'this is the rhythm'.split())

    def test_link_inline(self):
        if False:
            return 10
        ' m2m-style specification for o2ms\n        '
        id1 = self.env['export.one2many.child'].create({'str': 'Bf', 'value': 109}).id
        id2 = self.env['export.one2many.child'].create({'str': 'Me', 'value': 262}).id
        result = self.import_(['const', 'value/.id'], [['42', '%d,%d' % (id1, id2)]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        [b] = self.browse()
        self.assertEqual(b.const, 42)
        self.assertEqual(values(b.value), [109, 262])
        self.assertEqual(values(b.value, field='parent_id'), [b, b])

    def test_link(self):
        if False:
            return 10
        ' O2M relating to an existing record (update) force a LINK_TO as well\n        '
        id1 = self.env['export.one2many.child'].create({'str': 'Bf', 'value': 109}).id
        id2 = self.env['export.one2many.child'].create({'str': 'Me', 'value': 262}).id
        result = self.import_(['const', 'value/.id'], [['42', str(id1)], ['', str(id2)]])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        [b] = self.browse()
        self.assertEqual(b.const, 42)
        self.assertEqual(values(b.value), [109, 262])
        self.assertEqual(values(b.value, field='parent_id'), [b, b])

    def test_link_2(self):
        if False:
            for i in range(10):
                print('nop')
        id1 = self.env['export.one2many.child'].create({'str': 'Bf', 'value': 109}).id
        id2 = self.env['export.one2many.child'].create({'str': 'Me', 'value': 262}).id
        result = self.import_(['const', 'value/.id', 'value/value'], [['42', str(id1), '1'], ['', str(id2), '2']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        [b] = self.browse()
        self.assertEqual(b.const, 42)
        self.assertEqual(values(b.value), [1, 2])
        self.assertEqual(values(b.value, field='parent_id'), [b, b])

class test_o2m_multiple(ImporterCase):
    model_name = 'export.one2many.multiple'

    def test_multi_mixed(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.import_(['const', 'child1/value', 'child2/value'], [['5', '11', '21'], ['', '12', '22'], ['', '13', '23'], ['', '14', '']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        [b] = self.browse()
        self.assertEqual(values(b.child1), [11, 12, 13, 14])
        self.assertEqual(values(b.child2), [21, 22, 23])

    def test_multi(self):
        if False:
            while True:
                i = 10
        result = self.import_(['const', 'child1/value', 'child2/value'], [['5', '11', '21'], ['', '12', ''], ['', '13', ''], ['', '14', ''], ['', '', '22'], ['', '', '23']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        [b] = self.browse()
        self.assertEqual(values(b.child1), [11, 12, 13, 14])
        self.assertEqual(values(b.child2), [21, 22, 23])

    def test_multi_fullsplit(self):
        if False:
            print('Hello World!')
        result = self.import_(['const', 'child1/value', 'child2/value'], [['5', '11', ''], ['', '12', ''], ['', '13', ''], ['', '14', ''], ['', '', '21'], ['', '', '22'], ['', '', '23']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)
        [b] = self.browse()
        self.assertEqual(b.const, 5)
        self.assertEqual(values(b.child1), [11, 12, 13, 14])
        self.assertEqual(values(b.child2), [21, 22, 23])

class test_realworld(common.TransactionCase):

    def test_bigfile(self):
        if False:
            print('Hello World!')
        data = json.loads(pkgutil.get_data(self.__module__, 'contacts_big.json'))
        result = self.env['res.partner'].load(['name', 'mobile', 'email', 'image'], data)
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), len(data))

    def test_backlink(self):
        if False:
            return 10
        fnames = ['name', 'type', 'street', 'city', 'country_id', 'category_id', 'supplier', 'customer', 'is_company', 'parent_id']
        data = json.loads(pkgutil.get_data(self.__module__, 'contacts.json'))
        result = self.env['res.partner'].load(fnames, data)
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), len(data))

    def test_recursive_o2m(self):
        if False:
            i = 10
            return i + 15
        " The content of the o2m field's dict needs to go through conversion\n        as it may be composed of convertables or other relational fields\n        "
        self.env['ir.model.data'].clear_caches()
        Model = self.env['export.one2many.recursive']
        result = Model.load(['value', 'child/const', 'child/child1/str', 'child/child2/value'], [['4', '42', 'foo', '55'], ['', '43', 'bar', '56'], ['', '', 'baz', ''], ['', '55', 'qux', '57'], ['5', '99', 'wheee', ''], ['', '98', '', '12']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 2)
        b = Model.browse(result['ids'])
        self.assertEqual((b[0].value, b[1].value), (4, 5))
        self.assertEqual([child.str for child in b[0].child[1].child1], ['bar', 'baz'])
        self.assertFalse(len(b[1].child[1].child1))
        self.assertEqual([child.value for child in b[1].child[1].child2], [12])

class test_date(ImporterCase):
    model_name = 'export.date'

    def test_empty(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.import_(['value'], []), {'ids': [], 'messages': []})

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        result = self.import_(['value'], [['2012-02-03']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)

    def test_invalid(self):
        if False:
            return 10
        result = self.import_(['value'], [['not really a date']])
        self.assertEqual(result['messages'], [message(u"'not really a date' does not seem to be a valid date for field 'Value'", moreinfo=u"Use the format '2012-12-31'")])
        self.assertIs(result['ids'], False)

class test_datetime(ImporterCase):
    model_name = 'export.datetime'

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.import_(['value'], []), {'ids': [], 'messages': []})

    def test_basic(self):
        if False:
            print('Hello World!')
        result = self.import_(['value'], [['2012-02-03 11:11:11']])
        self.assertFalse(result['messages'])
        self.assertEqual(len(result['ids']), 1)

    def test_invalid(self):
        if False:
            print('Hello World!')
        result = self.import_(['value'], [['not really a datetime']])
        self.assertEqual(result['messages'], [message(u"'not really a datetime' does not seem to be a valid datetime for field 'Value'", moreinfo=u"Use the format '2012-12-31 23:59:59'")])
        self.assertIs(result['ids'], False)

    def test_checktz1(self):
        if False:
            for i in range(10):
                print('nop')
        ' Imported date should be interpreted as being in the tz provided by\n        the context\n        '
        self.env.user.write({'tz': 'Asia/Hovd'})
        result = self.import_(['value'], [['2012-02-03 11:11:11']], {'tz': 'Pacific/Kiritimati'})
        self.assertFalse(result['messages'])
        self.assertEqual(values(self.read(domain=[('id', 'in', result['ids'])])), ['2012-02-02 21:11:11'])
        result = self.import_(['value'], [['2012-02-03 11:11:11']], {'tz': 'Pacific/Marquesas'})
        self.assertFalse(result['messages'])
        self.assertEqual(values(self.read(domain=[('id', 'in', result['ids'])])), ['2012-02-03 20:41:11'])

    def test_usertz(self):
        if False:
            for i in range(10):
                print('nop')
        " If the context does not hold a timezone, the importing user's tz\n        should be used\n        "
        self.env.user.write({'tz': 'Asia/Yakutsk'})
        result = self.import_(['value'], [['2012-02-03 11:11:11']])
        self.assertFalse(result['messages'])
        self.assertEqual(values(self.read(domain=[('id', 'in', result['ids'])])), ['2012-02-03 01:11:11'])

    def test_notz(self):
        if False:
            return 10
        ' If there is no tz either in the context or on the user, falls back\n        to UTC\n        '
        self.env.user.write({'tz': False})
        result = self.import_(['value'], [['2012-02-03 11:11:11']])
        self.assertFalse(result['messages'])
        self.assertEqual(values(self.read(domain=[('id', 'in', result['ids'])])), ['2012-02-03 11:11:11'])

class test_unique(ImporterCase):
    model_name = 'export.unique'

    @mute_logger('odoo.sql_db')
    def test_unique(self):
        if False:
            return 10
        result = self.import_(['value'], [['1'], ['1'], ['2'], ['3'], ['3']])
        self.assertFalse(result['ids'])
        self.assertEqual(result['messages'], [dict(message=u"The value for the field 'value' already exists. This might be 'Value' in the current model, or a field of the same name in an o2m.", type='error', rows={'from': 1, 'to': 1}, record=1, field='value'), dict(message=u"The value for the field 'value' already exists. This might be 'Value' in the current model, or a field of the same name in an o2m.", type='error', rows={'from': 4, 'to': 4}, record=4, field='value')])
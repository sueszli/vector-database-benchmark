import textwrap
import unittest
from lxml import etree, html
from lxml.builder import E
from odoo.tests import common
from odoo.addons.web_editor.models.ir_qweb import html_to_text

class TestHTMLToText(unittest.TestCase):

    def test_rawstring(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('foobar', html_to_text(E.div('foobar')))

    def test_br(self):
        if False:
            return 10
        self.assertEqual('foo\nbar', html_to_text(E.div('foo', E.br(), 'bar')))
        self.assertEqual('foo\n\nbar\nbaz', html_to_text(E.div('foo', E.br(), E.br(), 'bar', E.br(), 'baz')))

    def test_p(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('foo\n\nbar\n\nbaz', html_to_text(E.div('foo', E.p('bar'), 'baz')))
        self.assertEqual('foo', html_to_text(E.div(E.p('foo'))))
        self.assertEqual('foo\n\nbar', html_to_text(E.div('foo', E.p('bar'))))
        self.assertEqual('foo\n\nbar', html_to_text(E.div(E.p('foo'), 'bar')))
        self.assertEqual('foo\n\nbar\n\nbaz', html_to_text(E.div(E.p('foo'), E.p('bar'), E.p('baz'))))

    def test_div(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('foo\nbar\nbaz', html_to_text(E.div('foo', E.div('bar'), 'baz')))
        self.assertEqual('foo', html_to_text(E.div(E.div('foo'))))
        self.assertEqual('foo\nbar', html_to_text(E.div('foo', E.div('bar'))))
        self.assertEqual('foo\nbar', html_to_text(E.div(E.div('foo'), 'bar')))
        self.assertEqual('foo\nbar\nbaz', html_to_text(E.div('foo', E.div('bar'), E.div('baz'))))

    def test_other_block(self):
        if False:
            while True:
                i = 10
        self.assertEqual('foo\nbar\nbaz', html_to_text(E.div('foo', E.section('bar'), 'baz')))

    def test_inline(self):
        if False:
            print('Hello World!')
        self.assertEqual('foobarbaz', html_to_text(E.div('foo', E.span('bar'), 'baz')))

    def test_whitespace(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('foo bar\nbaz', html_to_text(E.div('foo\nbar', E.br(), 'baz')))
        self.assertEqual('foo bar\nbaz', html_to_text(E.div(E.div(E.span('foo'), ' bar'), 'baz')))

class TestConvertBack(common.TransactionCase):

    def setUp(self):
        if False:
            return 10
        super(TestConvertBack, self).setUp()
        self.env = self.env(context={'inherit_branding': True})

    def field_rountrip_result(self, field, value, expected):
        if False:
            for i in range(10):
                print('nop')
        model = 'web_editor.converter.test'
        record = self.env[model].create({field: value})
        t = etree.Element('t')
        e = etree.Element('span')
        t.append(e)
        field_value = 'record.%s' % field
        e.set('t-field', field_value)
        rendered = self.env['ir.qweb'].render(t, {'record': record})
        element = html.fromstring(rendered, parser=html.HTMLParser(encoding='utf-8'))
        model = 'ir.qweb.field.' + element.get('data-oe-type', '')
        converter = self.env[model] if model in self.env else self.env['ir.qweb.field']
        value_back = converter.from_html(model, record._fields[field], element)
        if isinstance(expected, str):
            expected = expected.decode('utf-8')
        self.assertEqual(value_back, expected)

    def field_roundtrip(self, field, value):
        if False:
            while True:
                i = 10
        self.field_rountrip_result(field, value, value)

    def test_integer(self):
        if False:
            for i in range(10):
                print('nop')
        self.field_roundtrip('integer', 42)

    def test_float(self):
        if False:
            while True:
                i = 10
        self.field_roundtrip('float', 42.56789)
        self.field_roundtrip('float', 324542.56789)

    def test_numeric(self):
        if False:
            for i in range(10):
                print('nop')
        self.field_roundtrip('numeric', 42.77)

    def test_char(self):
        if False:
            while True:
                i = 10
        self.field_roundtrip('char', 'foo bar')
        self.field_roundtrip('char', 'ⒸⓄⓇⒼⒺ')

    def test_selection(self):
        if False:
            for i in range(10):
                print('nop')
        self.field_roundtrip('selection', 3)

    def test_selection_str(self):
        if False:
            while True:
                i = 10
        self.field_roundtrip('selection_str', 'B')

    def test_text(self):
        if False:
            while True:
                i = 10
        self.field_roundtrip('text', textwrap.dedent("            You must obey the dance commander\n            Givin' out the order for fun\n            You must obey the dance commander\n            You know that he's the only one\n            Who gives the orders here,\n            Alright\n            Who gives the orders here,\n            Alright\n\n            It would be awesome\n            If we could dance-a\n            It would be awesome, yeah\n            Let's take the chance-a\n            It would be awesome, yeah\n            Let's start the show\n            Because you never know\n            You never know\n            You never know until you go"))

    def test_m2o(self):
        if False:
            while True:
                i = 10
        ' the M2O field conversion (from html) is markedly different from\n        others as it directly writes into the m2o and returns nothing at all.\n        '
        field = 'many2one'
        subrec1 = self.env['web_editor.converter.test.sub'].create({'name': 'Foo'})
        subrec2 = self.env['web_editor.converter.test.sub'].create({'name': 'Bar'})
        record = self.env['web_editor.converter.test'].create({field: subrec1.id})
        t = etree.Element('t')
        e = etree.Element('span')
        t.append(e)
        field_value = 'record.%s' % field
        e.set('t-field', field_value)
        rendered = self.env['ir.qweb'].render(t, {'record': record})
        element = html.fromstring(rendered, parser=html.HTMLParser(encoding='utf-8'))
        element.set('data-oe-many2one-id', str(subrec2.id))
        element.text = 'New content'
        model = 'ir.qweb.field.' + element.get('data-oe-type')
        converter = self.env[model] if model in self.env else self.env['ir.qweb.field']
        value_back = converter.from_html('web_editor.converter.test', record._fields[field], element)
        self.assertIsNone(value_back, 'the m2o converter should return None to avoid spurious or useless writes on the parent record')
        self.assertEqual(subrec1.name, 'Foo', "element edition can't change directly the m2o record")
        self.assertEqual(record.many2one.name, 'Bar', 'element edition should have been change the m2o id')
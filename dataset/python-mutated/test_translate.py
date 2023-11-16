import unittest
from odoo.tools.translate import quote, unquote, xml_translate, html_translate
from odoo.tests.common import TransactionCase

class TranslationToolsTestCase(unittest.TestCase):

    def test_quote_unquote(self):
        if False:
            for i in range(10):
                print('nop')

        def test_string(str):
            if False:
                for i in range(10):
                    print('nop')
            quoted = quote(str)
            unquoted = unquote(''.join(quoted.split('"\n"')))
            self.assertEquals(str, unquoted)
        test_string('test \nall kinds\n \n o\r\n         \\\\ nope\n\n"\n         ')
        self.assertRaises(AssertionError, quote, 'test \nall kinds\n\no\r\n         \\\\nope\n\n"\n         ')

    def test_translate_xml_base(self):
        if False:
            while True:
                i = 10
        ' Test xml_translate() without formatting elements. '
        terms = []
        source = '<form string="Form stuff">\n                        <h1>Blah blah blah</h1>\n                        Put some more text here\n                        <field name="foo"/>\n                    </form>'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, ['Form stuff', 'Blah blah blah', 'Put some more text here'])

    def test_translate_xml_text(self):
        if False:
            return 10
        ' Test xml_translate() on plain text. '
        terms = []
        source = 'Blah blah blah'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, [source])

    def test_translate_xml_text_entity(self):
        if False:
            i = 10
            return i + 15
        ' Test xml_translate() on plain text with HTML escaped entities. '
        terms = []
        source = 'Blah&amp;nbsp;blah&amp;nbsp;blah'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, [source])

    def test_translate_xml_inline1(self):
        if False:
            print('Hello World!')
        ' Test xml_translate() with formatting elements. '
        terms = []
        source = '<form string="Form stuff">\n                        <h1>Blah <i>blah</i> blah</h1>\n                        Put some <b>more text</b> here\n                        <field name="foo"/>\n                    </form>'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, ['Form stuff', 'Blah <i>blah</i> blah', 'Put some <b>more text</b> here'])

    def test_translate_xml_inline2(self):
        if False:
            while True:
                i = 10
        ' Test xml_translate() with formatting elements embedding other elements. '
        terms = []
        source = '<form string="Form stuff">\n                        <b><h1>Blah <i>blah</i> blah</h1></b>\n                        Put <em>some <b>more text</b></em> here\n                        <field name="foo"/>\n                    </form>'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, ['Form stuff', 'Blah <i>blah</i> blah', 'Put <em>some <b>more text</b></em> here'])

    def test_translate_xml_inline3(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test xml_translate() with formatting elements without actual text. '
        terms = []
        source = '<form string="Form stuff">\n                        <div>\n                            <span class="before"/>\n                            <h1>Blah blah blah</h1>\n                            <span class="after">\n                                <i class="hack"/>\n                            </span>\n                        </div>\n                    </form>'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, ['Form stuff', 'Blah blah blah'])

    def test_translate_xml_t(self):
        if False:
            return 10
        ' Test xml_translate() with t-* attributes. '
        terms = []
        source = '<t t-name="stuff">\n                        stuff before\n                        <span t-field="o.name"/>\n                        stuff after\n                    </t>'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, ['stuff before', 'stuff after'])

    def test_translate_xml_off(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test xml_translate() with attribute translate="off". '
        terms = []
        source = '<div>\n                        stuff before\n                        <div t-translation="off">Do not translate this</div>\n                        stuff after\n                    </div>'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, ['stuff before', 'stuff after'])

    def test_translate_xml_attribute(self):
        if False:
            while True:
                i = 10
        ' Test xml_translate() with <attribute> elements. '
        terms = []
        source = '<field name="foo" position="attributes">\n                        <attribute name="string">Translate this</attribute>\n                        <attribute name="option">Do not translate this</attribute>\n                    </field>'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, ['Translate this'])

    def test_translate_xml_a(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test xml_translate() with <a> elements. '
        terms = []
        source = '<t t-name="stuff">\n                        <ul class="nav navbar-nav">\n                            <li>\n                                <a class="oe_menu_leaf" href="/web#menu_id=42&amp;action=54">\n                                    <span class="oe_menu_text">Blah</span>\n                                </a>\n                            </li>\n                            <li class="dropdown" id="menu_more_container" style="display: none;">\n                                <a class="dropdown-toggle" data-toggle="dropdown" href="#">More <b class="caret"/></a>\n                                <ul class="dropdown-menu" id="menu_more"/>\n                            </li>\n                        </ul>\n                    </t>'
        result = xml_translate(terms.append, source)
        self.assertEquals(result, source)
        self.assertItemsEqual(terms, ['<span class="oe_menu_text">Blah</span>', 'More <b class="caret"/>'])

    def test_translate_html(self):
        if False:
            while True:
                i = 10
        ' Test xml_translate() and html_translate() with <i> elements. '
        source = '<i class="fa-check"></i>'
        result = xml_translate(lambda term: term, source)
        self.assertEquals(result, '<i class="fa-check"/>')
        result = html_translate(lambda term: term, source)
        self.assertEquals(result, source)

class TestTranslation(TransactionCase):

    def setUp(self):
        if False:
            return 10
        super(TestTranslation, self).setUp()
        self.env['ir.translation'].load_module_terms(['base'], ['fr_FR'])
        self.customers = self.env['res.partner.category'].create({'name': 'Customers'})
        self.env['ir.translation'].create({'type': 'model', 'name': 'res.partner.category,name', 'module': 'base', 'lang': 'fr_FR', 'res_id': self.customers.id, 'value': 'Clients', 'state': 'translated'})

    def test_101_create_translated_record(self):
        if False:
            for i in range(10):
                print('nop')
        category = self.customers.with_context({})
        self.assertEqual(category.name, 'Customers', 'Error in basic name_get')
        category_fr = category.with_context({'lang': 'fr_FR'})
        self.assertEqual(category_fr.name, 'Clients', 'Translation not found')

    def test_102_duplicate_record(self):
        if False:
            for i in range(10):
                print('nop')
        category = self.customers.with_context({'lang': 'fr_FR'}).copy()
        category_no = category.with_context({})
        self.assertEqual(category_no.name, 'Customers', 'Duplication did not set untranslated value')
        category_fr = category.with_context({'lang': 'fr_FR'})
        self.assertEqual(category_fr.name, 'Clients', 'Did not found translation for initial value')

    def test_103_duplicate_record_fr(self):
        if False:
            print('Hello World!')
        category = self.customers.with_context({'lang': 'fr_FR'}).copy({'name': 'Clients (copie)'})
        category_no = category.with_context({})
        self.assertEqual(category_no.name, 'Customers', 'Duplication erased original untranslated value')
        category_fr = category.with_context({'lang': 'fr_FR'})
        self.assertEqual(category_fr.name, 'Clients (copie)', 'Did not used default value for translated value')

    def test_104_orderby_translated_field(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test search ordered by a translated field. '
        padawans = self.env['res.partner.category'].create({'name': 'Padawans'})
        padawans_fr = padawans.with_context(lang='fr_FR')
        padawans_fr.write({'name': 'Apprentis'})
        categories = padawans_fr.search([('id', 'in', [self.customers.id, padawans.id])], order='name')
        self.assertEqual(categories.ids, [padawans.id, self.customers.id], 'Search ordered by translated name should return Padawans (Apprentis) before Customers (Clients)')

class TestXMLTranslation(TransactionCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestXMLTranslation, self).setUp()
        self.env['ir.translation'].load_module_terms(['base'], ['fr_FR'])

    def test_copy(self):
        if False:
            for i in range(10):
                print('nop')
        ' Create a simple view, fill in translations, and copy it. '
        env_en = self.env(context={})
        env_fr = self.env(context={'lang': 'fr_FR'})
        archf = '<form string="%s"><div>%s</div><div>%s</div></form>'
        terms_en = ('Knife', 'Fork', 'Spoon')
        terms_fr = ('Couteau', 'Fourchette', 'Cuiller')
        view0 = self.env['ir.ui.view'].create({'name': 'test', 'model': 'res.partner', 'arch': archf % terms_en})
        for (src, value) in zip(terms_en, terms_fr):
            self.env['ir.translation'].create({'type': 'model', 'name': 'ir.ui.view,arch_db', 'lang': 'fr_FR', 'res_id': view0.id, 'src': src, 'value': value})
        self.assertEqual(view0.with_env(env_en).arch_db, archf % terms_en)
        self.assertEqual(view0.with_env(env_fr).arch_db, archf % terms_fr)
        view1 = view0.with_env(env_en).copy({})
        self.assertEqual(view1.with_env(env_en).arch_db, archf % terms_en)
        self.assertEqual(view1.with_env(env_fr).arch_db, archf % terms_fr)
        view2 = view0.with_env(env_fr).copy({})
        self.assertEqual(view2.with_env(env_en).arch_db, archf % terms_en)
        self.assertEqual(view2.with_env(env_fr).arch_db, archf % terms_fr)
        self.patch(type(self.env['ir.ui.view']).arch_db, 'translate', html_translate)
        view3 = view0.with_env(env_fr).copy({})
        self.assertEqual(view3.with_env(env_en).arch_db, archf % terms_en)
        self.assertEqual(view3.with_env(env_fr).arch_db, archf % terms_fr)

    def test_spaces(self):
        if False:
            print('Hello World!')
        ' Create translations where value has surrounding spaces. '
        archf = '<form string="%s"><div>%s</div><div>%s</div></form>'
        terms_en = ('Knife', 'Fork', 'Spoon')
        terms_fr = (' Couteau', 'Fourchette ', ' Cuiller ')
        view0 = self.env['ir.ui.view'].create({'name': 'test', 'model': 'res.partner', 'arch': archf % terms_en})
        for (src, value) in zip(terms_en, terms_fr):
            self.env['ir.translation'].create({'type': 'model', 'name': 'ir.ui.view,arch_db', 'lang': 'fr_FR', 'res_id': view0.id, 'src': src, 'value': value})
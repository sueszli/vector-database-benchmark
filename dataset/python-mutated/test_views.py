from functools import partial
from itertools import izip_longest
from lxml import etree
from lxml.builder import E
from psycopg2 import IntegrityError
from odoo.osv.orm import modifiers_tests
from odoo.exceptions import ValidationError
from odoo.tests import common
from odoo.tools import mute_logger

class ViewXMLID(common.TransactionCase):

    def test_model_data_id(self):
        if False:
            for i in range(10):
                print('nop')
        ' Check whether views know their xmlid record. '
        view = self.env.ref('base.view_company_form')
        self.assertTrue(view)
        self.assertTrue(view.model_data_id)
        self.assertEqual(view.model_data_id.complete_name, 'base.view_company_form')

class ViewCase(common.TransactionCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(ViewCase, self).setUp()
        self.addTypeEqualityFunc(etree._Element, self.assertTreesEqual)
        self.View = self.env['ir.ui.view']

    def assertTreesEqual(self, n1, n2, msg=None):
        if False:
            while True:
                i = 10
        self.assertEqual(n1.tag, n2.tag, msg)
        self.assertEqual((n1.text or '').strip(), (n2.text or '').strip(), msg)
        self.assertEqual((n1.tail or '').strip(), (n2.tail or '').strip(), msg)
        self.assertEqual(dict(n1.attrib), dict(n2.attrib), msg)
        for (c1, c2) in izip_longest(n1, n2):
            self.assertEqual(c1, c2, msg)

class TestNodeLocator(common.TransactionCase):
    """
    The node locator returns None when it can not find a node, and the first
    match when it finds something (no jquery-style node sets)
    """

    def test_no_match_xpath(self):
        if False:
            return 10
        '\n        xpath simply uses the provided @expr pattern to find a node\n        '
        node = self.env['ir.ui.view'].locate_node(E.root(E.foo(), E.bar(), E.baz()), E.xpath(expr='//qux'))
        self.assertIsNone(node)

    def test_match_xpath(self):
        if False:
            while True:
                i = 10
        bar = E.bar()
        node = self.env['ir.ui.view'].locate_node(E.root(E.foo(), bar, E.baz()), E.xpath(expr='//bar'))
        self.assertIs(node, bar)

    def test_no_match_field(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A field spec will match by @name against all fields of the view\n        '
        node = self.env['ir.ui.view'].locate_node(E.root(E.foo(), E.bar(), E.baz()), E.field(name='qux'))
        self.assertIsNone(node)
        node = self.env['ir.ui.view'].locate_node(E.root(E.field(name='foo'), E.field(name='bar'), E.field(name='baz')), E.field(name='qux'))
        self.assertIsNone(node)

    def test_match_field(self):
        if False:
            return 10
        bar = E.field(name='bar')
        node = self.env['ir.ui.view'].locate_node(E.root(E.field(name='foo'), bar, E.field(name='baz')), E.field(name='bar'))
        self.assertIs(node, bar)

    def test_no_match_other(self):
        if False:
            return 10
        '\n        Non-xpath non-fields are matched by node name first\n        '
        node = self.env['ir.ui.view'].locate_node(E.root(E.foo(), E.bar(), E.baz()), E.qux())
        self.assertIsNone(node)

    def test_match_other(self):
        if False:
            for i in range(10):
                print('nop')
        bar = E.bar()
        node = self.env['ir.ui.view'].locate_node(E.root(E.foo(), bar, E.baz()), E.bar())
        self.assertIs(bar, node)

    def test_attribute_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Non-xpath non-field are filtered by matching attributes on spec and\n        matched nodes\n        '
        node = self.env['ir.ui.view'].locate_node(E.root(E.foo(attr='1'), E.bar(attr='2'), E.baz(attr='3')), E.bar(attr='5'))
        self.assertIsNone(node)

    def test_attribute_filter(self):
        if False:
            i = 10
            return i + 15
        match = E.bar(attr='2')
        node = self.env['ir.ui.view'].locate_node(E.root(E.bar(attr='1'), match, E.root(E.bar(attr='3'))), E.bar(attr='2'))
        self.assertIs(node, match)

    def test_version_mismatch(self):
        if False:
            while True:
                i = 10
        "\n        A @version on the spec will be matched against the view's version\n        "
        node = self.env['ir.ui.view'].locate_node(E.root(E.foo(attr='1'), version='4'), E.foo(attr='1', version='3'))
        self.assertIsNone(node)

class TestViewInheritance(ViewCase):

    def arch_for(self, name, view_type='form', parent=None):
        if False:
            print('Hello World!')
        " Generates a trivial view of the specified ``view_type``.\n\n        The generated view is empty but ``name`` is set as its root's ``@string``.\n\n        If ``parent`` is not falsy, generates an extension view (instead of\n        a root view) replacing the parent's ``@string`` by ``name``\n\n        :param str name: ``@string`` value for the view root\n        :param str view_type:\n        :param bool parent:\n        :return: generated arch\n        :rtype: str\n        "
        if not parent:
            element = E(view_type, string=name)
        else:
            element = E(view_type, E.attribute(name, name='string'), position='attributes')
        return etree.tostring(element)

    def makeView(self, name, parent=None, arch=None):
        if False:
            print('Hello World!')
        " Generates a basic ir.ui.view with the provided name, parent and arch.\n\n        If no parent is provided, the view is top-level.\n\n        If no arch is provided, generates one by calling :meth:`~.arch_for`.\n\n        :param str name:\n        :param int parent: id of the parent view, if any\n        :param str arch:\n        :returns: the created view's id.\n        :rtype: int\n        "
        view = self.View.create({'model': self.model, 'name': name, 'arch': arch or self.arch_for(name, parent=parent), 'inherit_id': parent, 'priority': 5})
        self.view_ids[name] = view.id
        return view

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestViewInheritance, self).setUp()
        self.patch(self.registry, '_init', False)
        self.model = 'ir.ui.view.custom'
        self.view_ids = {}
        a = self.makeView('A')
        a1 = self.makeView('A1', a.id)
        a11 = self.makeView('A11', a1.id)
        self.makeView('A111', a11.id)
        self.makeView('A12', a1.id)
        a2 = self.makeView('A2', a.id)
        self.makeView('A21', a2.id)
        a22 = self.makeView('A22', a2.id)
        self.makeView('A221', a22.id)
        b = self.makeView('B', arch=self.arch_for('B', 'tree'))
        self.makeView('B1', b.id, arch=self.arch_for('B1', 'tree', parent=b))
        c = self.makeView('C', arch=self.arch_for('C', 'tree'))
        c.write({'priority': 1})

    def test_get_inheriting_views_arch(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.View.get_inheriting_views_arch(self.view_ids['A'], self.model), [(self.arch_for('A1', parent=True), self.view_ids['A1']), (self.arch_for('A2', parent=True), self.view_ids['A2'])])
        self.assertEqual(self.View.get_inheriting_views_arch(self.view_ids['A21'], self.model), [])
        self.assertEqual(self.View.get_inheriting_views_arch(self.view_ids['A11'], self.model), [(self.arch_for('A111', parent=True), self.view_ids['A111'])])

    def test_default_view(self):
        if False:
            for i in range(10):
                print('nop')
        default = self.View.default_view(model=self.model, view_type='form')
        self.assertEqual(default, self.view_ids['A'])
        default_tree = self.View.default_view(model=self.model, view_type='tree')
        self.assertEqual(default_tree, self.view_ids['C'])

    def test_no_default_view(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self.View.default_view(model='does.not.exist', view_type='form'))
        self.assertFalse(self.View.default_view(model=self.model, view_type='graph'))

    def test_no_recursion(self):
        if False:
            print('Hello World!')
        r1 = self.makeView('R1')
        with self.assertRaises(ValidationError), self.cr.savepoint():
            r1.write({'inherit_id': r1.id})
        r2 = self.makeView('R2', r1.id)
        r3 = self.makeView('R3', r2.id)
        with self.assertRaises(ValidationError), self.cr.savepoint():
            r2.write({'inherit_id': r3.id})
        with self.assertRaises(ValidationError), self.cr.savepoint():
            r1.write({'inherit_id': r3.id})
        with self.assertRaises(ValidationError), self.cr.savepoint():
            r1.write({'inherit_id': r1.id, 'arch': self.arch_for('itself', parent=True)})

class TestApplyInheritanceSpecs(ViewCase):
    """ Applies a sequence of inheritance specification nodes to a base
    architecture. IO state parameters (cr, uid, model, context) are used for
    error reporting

    The base architecture is altered in-place.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestApplyInheritanceSpecs, self).setUp()
        self.base_arch = E.form(E.field(name='target'), string='Title')

    def test_replace(self):
        if False:
            while True:
                i = 10
        spec = E.field(E.field(name='replacement'), name='target', position='replace')
        self.View.apply_inheritance_specs(self.base_arch, spec, None)
        self.assertEqual(self.base_arch, E.form(E.field(name='replacement'), string='Title'))

    def test_delete(self):
        if False:
            print('Hello World!')
        spec = E.field(name='target', position='replace')
        self.View.apply_inheritance_specs(self.base_arch, spec, None)
        self.assertEqual(self.base_arch, E.form(string='Title'))

    def test_insert_after(self):
        if False:
            while True:
                i = 10
        spec = E.field(E.field(name='inserted'), name='target', position='after')
        self.View.apply_inheritance_specs(self.base_arch, spec, None)
        self.assertEqual(self.base_arch, E.form(E.field(name='target'), E.field(name='inserted'), string='Title'))

    def test_insert_before(self):
        if False:
            print('Hello World!')
        spec = E.field(E.field(name='inserted'), name='target', position='before')
        self.View.apply_inheritance_specs(self.base_arch, spec, None)
        self.assertEqual(self.base_arch, E.form(E.field(name='inserted'), E.field(name='target'), string='Title'))

    def test_insert_inside(self):
        if False:
            for i in range(10):
                print('nop')
        default = E.field(E.field(name='inserted'), name='target')
        spec = E.field(E.field(name='inserted 2'), name='target', position='inside')
        self.View.apply_inheritance_specs(self.base_arch, default, None)
        self.View.apply_inheritance_specs(self.base_arch, spec, None)
        self.assertEqual(self.base_arch, E.form(E.field(E.field(name='inserted'), E.field(name='inserted 2'), name='target'), string='Title'))

    def test_unpack_data(self):
        if False:
            print('Hello World!')
        spec = E.data(E.field(E.field(name='inserted 0'), name='target'), E.field(E.field(name='inserted 1'), name='target'), E.field(E.field(name='inserted 2'), name='target'), E.field(E.field(name='inserted 3'), name='target'))
        self.View.apply_inheritance_specs(self.base_arch, spec, None)
        self.assertEqual(self.base_arch, E.form(E.field(E.field(name='inserted 0'), E.field(name='inserted 1'), E.field(name='inserted 2'), E.field(name='inserted 3'), name='target'), string='Title'))

    @mute_logger('odoo.addons.base.ir.ir_ui_view')
    def test_invalid_position(self):
        if False:
            while True:
                i = 10
        spec = E.field(E.field(name='whoops'), name='target', position='serious_series')
        with self.assertRaises(ValueError):
            self.View.apply_inheritance_specs(self.base_arch, spec, None)

    @mute_logger('odoo.addons.base.ir.ir_ui_view')
    def test_incorrect_version(self):
        if False:
            for i in range(10):
                print('nop')
        arch = E.form(E.element(foo='42'))
        spec = E.element(E.field(name='placeholder'), foo='42', version='7.0')
        with self.assertRaises(ValueError):
            self.View.apply_inheritance_specs(arch, spec, None)

    @mute_logger('odoo.addons.base.ir.ir_ui_view')
    def test_target_not_found(self):
        if False:
            while True:
                i = 10
        spec = E.field(name='targut')
        with self.assertRaises(ValueError):
            self.View.apply_inheritance_specs(self.base_arch, spec, None)

class TestApplyInheritanceWrapSpecs(ViewCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestApplyInheritanceWrapSpecs, self).setUp()
        self.base_arch = E.template(E.div(E.p('Content')))

    def apply_spec(self, spec):
        if False:
            return 10
        self.View.apply_inheritance_specs(self.base_arch, spec, None)

    def test_replace(self):
        if False:
            for i in range(10):
                print('nop')
        spec = E.xpath(E.div('$0', {'class': 'some'}), expr='//p', position='replace')
        self.apply_spec(spec)
        self.assertEqual(self.base_arch, E.template(E.div(E.div(E.p('Content'), {'class': 'some'}))))

class TestApplyInheritedArchs(ViewCase):
    """ Applies a sequence of modificator archs to a base view
    """

class TestNoModel(ViewCase):

    def test_create_view_nomodel(self):
        if False:
            return 10
        view = self.View.create({'name': 'dummy', 'arch': '<template name="foo"/>', 'inherit_id': False, 'type': 'qweb'})
        fields = ['name', 'arch', 'type', 'priority', 'inherit_id', 'model']
        [data] = view.read(fields)
        self.assertEqual(data, {'id': view.id, 'name': 'dummy', 'arch': '<template name="foo"/>', 'type': 'qweb', 'priority': 16, 'inherit_id': False, 'model': False})
    text_para = E.p('', {'class': 'legalese'})
    arch = E.body(E.div(E.h1('Title'), id='header'), E.p('Welcome!'), E.div(E.hr(), text_para, id='footer'), {'class': 'index'})

    def test_qweb_translation(self):
        if False:
            while True:
                i = 10
        '\n        Test if translations work correctly without a model\n        '
        self.env['res.lang'].load_lang('fr_FR')
        ARCH = '<template name="foo">%s</template>'
        TEXT_EN = 'Copyright copyrighter'
        TEXT_FR = u'Copyrighter, tous droits réservés'
        view = self.View.create({'name': 'dummy', 'arch': ARCH % TEXT_EN, 'inherit_id': False, 'type': 'qweb'})
        self.env['ir.translation'].create({'type': 'model', 'name': 'ir.ui.view,arch_db', 'res_id': view.id, 'lang': 'fr_FR', 'src': TEXT_EN, 'value': TEXT_FR})
        view = view.with_context(lang='fr_FR')
        self.assertEqual(view.arch, ARCH % TEXT_FR)

class TestTemplating(ViewCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestTemplating, self).setUp()
        self.patch(self.registry, '_init', False)

    def test_branding_inherit(self):
        if False:
            return 10
        view1 = self.View.create({'name': 'Base view', 'type': 'qweb', 'arch': '<root>\n                <item order="1"/>\n            </root>\n            '})
        view2 = self.View.create({'name': 'Extension', 'type': 'qweb', 'inherit_id': view1.id, 'arch': '<xpath expr="//item" position="before">\n                <item order="2"/>\n            </xpath>\n            '})
        arch_string = view1.with_context(inherit_branding=True).read_combined(['arch'])['arch']
        arch = etree.fromstring(arch_string)
        self.View.distribute_branding(arch)
        [initial] = arch.xpath('//item[@order=1]')
        self.assertEqual(str(view1.id), initial.get('data-oe-id'), 'initial should come from the root view')
        self.assertEqual('/root[1]/item[1]', initial.get('data-oe-xpath'), "initial's xpath should be within the root view only")
        [second] = arch.xpath('//item[@order=2]')
        self.assertEqual(str(view2.id), second.get('data-oe-id'), 'second should come from the extension view')

    def test_branding_distribute_inner(self):
        if False:
            return 10
        ' Checks that the branding is correctly distributed within a view\n        extension\n        '
        view1 = self.View.create({'name': 'Base view', 'type': 'qweb', 'arch': '<root>\n                <item order="1"/>\n            </root>'})
        view2 = self.View.create({'name': 'Extension', 'type': 'qweb', 'inherit_id': view1.id, 'arch': '<xpath expr="//item" position="before">\n                <item order="2">\n                    <content t-att-href="foo">bar</content>\n                </item>\n            </xpath>'})
        arch_string = view1.with_context(inherit_branding=True).read_combined(['arch'])['arch']
        arch = etree.fromstring(arch_string)
        self.View.distribute_branding(arch)
        self.assertEqual(arch, E.root(E.item(E.content('bar', {'t-att-href': 'foo', 'data-oe-model': 'ir.ui.view', 'data-oe-id': str(view2.id), 'data-oe-field': 'arch', 'data-oe-xpath': '/xpath/item/content[1]'}), {'order': '2'}), E.item({'order': '1', 'data-oe-model': 'ir.ui.view', 'data-oe-id': str(view1.id), 'data-oe-field': 'arch', 'data-oe-xpath': '/root[1]/item[1]'})))

    def test_esc_no_branding(self):
        if False:
            while True:
                i = 10
        view = self.View.create({'name': 'Base View', 'type': 'qweb', 'arch': '<root>\n                <item><span t-esc="foo"/></item>\n            </root>'})
        arch_string = view.with_context(inherit_branding=True).read_combined(['arch'])['arch']
        arch = etree.fromstring(arch_string)
        self.View.distribute_branding(arch)
        self.assertEqual(arch, E.root(E.item(E.span({'t-esc': 'foo'}))))

    def test_ignore_unbrand(self):
        if False:
            while True:
                i = 10
        view1 = self.View.create({'name': 'Base view', 'type': 'qweb', 'arch': '<root>\n                <item order="1" t-ignore="true">\n                    <t t-esc="foo"/>\n                </item>\n            </root>'})
        view2 = self.View.create({'name': 'Extension', 'type': 'qweb', 'inherit_id': view1.id, 'arch': '<xpath expr="//item[@order=\'1\']" position="inside">\n                <item order="2">\n                    <content t-att-href="foo">bar</content>\n                </item>\n            </xpath>'})
        arch_string = view1.with_context(inherit_branding=True).read_combined(['arch'])['arch']
        arch = etree.fromstring(arch_string)
        self.View.distribute_branding(arch)
        self.assertEqual(arch, E.root(E.item({'t-ignore': 'true', 'order': '1'}, E.t({'t-esc': 'foo'}), E.item({'order': '2'}, E.content({'t-att-href': 'foo'}, 'bar')))), "t-ignore should apply to injected sub-view branding, not just to the main view's")

class TestViews(ViewCase):

    def test_nonexistent_attribute_removal(self):
        if False:
            for i in range(10):
                print('nop')
        self.View.create({'name': 'Test View', 'model': 'ir.ui.view', 'inherit_id': self.ref('base.view_view_tree'), 'arch': '<?xml version="1.0"?>\n                        <xpath expr="//field[@name=\'name\']" position="attributes">\n                            <attribute name="non_existing_attribute"></attribute>\n                        </xpath>\n                    '})

    def _insert_view(self, **kw):
        if False:
            while True:
                i = 10
        'Insert view into database via a query to passtrough validation'
        kw.pop('id', None)
        kw.setdefault('mode', 'extension' if kw.get('inherit_id') else 'primary')
        kw.setdefault('active', True)
        keys = sorted(kw.keys())
        fields = ','.join(('"%s"' % (k.replace('"', '\\"'),) for k in keys))
        params = ','.join(('%%(%s)s' % (k,) for k in keys))
        query = 'INSERT INTO ir_ui_view(%s) VALUES(%s) RETURNING id' % (fields, params)
        self.cr.execute(query, kw)
        return self.cr.fetchone()[0]

    def test_custom_view_validation(self):
        if False:
            while True:
                i = 10
        model = 'ir.actions.act_url'
        validate = partial(self.View._validate_custom_views, model)
        vid = self._insert_view(name='base view', model=model, priority=1, arch_db='<?xml version="1.0"?>\n                        <tree string="view">\n                          <field name="url"/>\n                        </tree>\n                    ')
        self.assertTrue(validate())
        self._insert_view(name='inherited view', model=model, priority=1, inherit_id=vid, arch_db='<?xml version="1.0"?>\n                        <xpath expr="//field[@name=\'url\']" position="before">\n                          <field name="name"/>\n                        </xpath>\n                    ')
        self.assertTrue(validate())
        self._insert_view(name='inherited view 2', model=model, priority=5, inherit_id=vid, arch_db='<?xml version="1.0"?>\n                        <xpath expr="//field[@name=\'name\']" position="after">\n                          <field name="target"/>\n                        </xpath>\n                    ')
        self.assertTrue(validate())

    def test_view_inheritance(self):
        if False:
            i = 10
            return i + 15
        view1 = self.View.create({'name': 'bob', 'model': 'ir.ui.view', 'arch': '\n                <form string="Base title" version="7.0">\n                    <separator name="separator" string="Separator" colspan="4"/>\n                    <footer>\n                        <button name="action_next" type="object" string="Next button" class="btn-primary"/>\n                        <button string="Skip" special="cancel" class="btn-default"/>\n                    </footer>\n                </form>\n            '})
        view2 = self.View.create({'name': 'edmund', 'model': 'ir.ui.view', 'inherit_id': view1.id, 'arch': '\n                <data>\n                    <form position="attributes" version="7.0">\n                        <attribute name="string">Replacement title</attribute>\n                    </form>\n                    <footer position="replace">\n                        <footer>\n                            <button name="action_next" type="object" string="New button"/>\n                        </footer>\n                    </footer>\n                    <separator name="separator" position="replace">\n                        <p>Replacement data</p>\n                    </separator>\n                </data>\n            '})
        view3 = self.View.create({'name': 'jake', 'model': 'ir.ui.view', 'inherit_id': view1.id, 'priority': 17, 'arch': '\n                <footer position="attributes">\n                    <attribute name="thing">bob tata lolo</attribute>\n                    <attribute name="thing" add="bibi and co" remove="tata" separator=" " />\n                    <attribute name="otherthing">bob, tata,lolo</attribute>\n                    <attribute name="otherthing" remove="tata, bob"/>\n                </footer>\n            '})
        view = self.View.with_context(check_view_ids=[view2.id, view3.id]).fields_view_get(view2.id, view_type='form')
        self.assertEqual(view['type'], 'form')
        self.assertEqual(etree.fromstring(view['arch'], parser=etree.XMLParser(remove_blank_text=True)), E.form(E.p('Replacement data'), E.footer(E.button(name='action_next', type='object', string='New button'), thing='bob lolo bibi and co', otherthing='lolo'), string='Replacement title', version='7.0'))

    def test_view_inheritance_divergent_models(self):
        if False:
            for i in range(10):
                print('nop')
        view1 = self.View.create({'name': 'bob', 'model': 'ir.ui.view.custom', 'arch': '\n                <form string="Base title" version="7.0">\n                    <separator name="separator" string="Separator" colspan="4"/>\n                    <footer>\n                        <button name="action_next" type="object" string="Next button" class="btn-primary"/>\n                        <button string="Skip" special="cancel" class="btn-default"/>\n                    </footer>\n                </form>\n            '})
        view2 = self.View.create({'name': 'edmund', 'model': 'ir.ui.view', 'inherit_id': view1.id, 'arch': '\n                <data>\n                    <form position="attributes" version="7.0">\n                        <attribute name="string">Replacement title</attribute>\n                    </form>\n                    <footer position="replace">\n                        <footer>\n                            <button name="action_next" type="object" string="New button"/>\n                        </footer>\n                    </footer>\n                    <separator name="separator" position="replace">\n                        <p>Replacement data</p>\n                    </separator>\n                </data>\n            '})
        view3 = self.View.create({'name': 'jake', 'model': 'ir.ui.menu', 'inherit_id': view1.id, 'priority': 17, 'arch': '\n                <footer position="attributes">\n                    <attribute name="thing">bob</attribute>\n                </footer>\n            '})
        view = self.View.with_context(check_view_ids=[view2.id, view3.id]).fields_view_get(view2.id, view_type='form')
        self.assertEqual(view['type'], 'form')
        self.assertEqual(etree.fromstring(view['arch'], parser=etree.XMLParser(remove_blank_text=True)), E.form(E.p('Replacement data'), E.footer(E.button(name='action_next', type='object', string='New button')), string='Replacement title', version='7.0'))

    def test_modifiers(self):
        if False:
            for i in range(10):
                print('nop')
        modifiers_tests()

class ViewModeField(ViewCase):
    """
    This should probably, eventually, be folded back into other test case
    classes, integrating the test (or not) of the mode field to regular cases
    """

    def testModeImplicitValue(self):
        if False:
            i = 10
            return i + 15
        ' mode is auto-generated from inherit_id:\n        * inherit_id -> mode=extension\n        * not inherit_id -> mode=primary\n        '
        view = self.View.create({'inherit_id': None, 'arch': '<qweb/>'})
        self.assertEqual(view.mode, 'primary')
        view2 = self.View.create({'inherit_id': view.id, 'arch': '<qweb/>'})
        self.assertEqual(view2.mode, 'extension')

    @mute_logger('odoo.sql_db')
    def testModeExplicit(self):
        if False:
            for i in range(10):
                print('nop')
        view = self.View.create({'inherit_id': None, 'arch': '<qweb/>'})
        view2 = self.View.create({'inherit_id': view.id, 'mode': 'primary', 'arch': '<qweb/>'})
        self.assertEqual(view.mode, 'primary')
        with self.assertRaises(IntegrityError):
            self.View.create({'inherit_id': None, 'mode': 'extension', 'arch': '<qweb/>'})

    @mute_logger('odoo.sql_db')
    def testPurePrimaryToExtension(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        A primary view with inherit_id=None can't be converted to extension\n        "
        view_pure_primary = self.View.create({'inherit_id': None, 'arch': '<qweb/>'})
        with self.assertRaises(IntegrityError):
            view_pure_primary.write({'mode': 'extension'})

    def testInheritPrimaryToExtension(self):
        if False:
            while True:
                i = 10
        '\n        A primary view with an inherit_id can be converted to extension\n        '
        base = self.View.create({'inherit_id': None, 'arch': '<qweb/>'})
        view = self.View.create({'inherit_id': base.id, 'mode': 'primary', 'arch': '<qweb/>'})
        view.write({'mode': 'extension'})

    def testDefaultExtensionToPrimary(self):
        if False:
            while True:
                i = 10
        '\n        An extension view can be converted to primary\n        '
        base = self.View.create({'inherit_id': None, 'arch': '<qweb/>'})
        view = self.View.create({'inherit_id': base.id, 'arch': '<qweb/>'})
        view.write({'mode': 'primary'})

class TestDefaultView(ViewCase):

    def testDefaultViewBase(self):
        if False:
            for i in range(10):
                print('nop')
        self.View.create({'inherit_id': False, 'priority': 10, 'mode': 'primary', 'arch': '<qweb/>'})
        view2 = self.View.create({'inherit_id': False, 'priority': 1, 'mode': 'primary', 'arch': '<qweb/>'})
        default = self.View.default_view(False, 'qweb')
        self.assertEqual(default, view2.id, 'default_view should get the view with the lowest priority for a (model, view_type) pair')

    def testDefaultViewPrimary(self):
        if False:
            return 10
        view1 = self.View.create({'inherit_id': False, 'priority': 10, 'mode': 'primary', 'arch': '<qweb/>'})
        self.View.create({'inherit_id': False, 'priority': 5, 'mode': 'primary', 'arch': '<qweb/>'})
        view3 = self.View.create({'inherit_id': view1.id, 'priority': 1, 'mode': 'primary', 'arch': '<qweb/>'})
        default = self.View.default_view(False, 'qweb')
        self.assertEqual(default, view3.id, 'default_view should get the view with the lowest priority for a (model, view_type) pair in all the primary tables')

class TestViewCombined(ViewCase):
    """
    * When asked for a view, instead of looking for the closest parent with
      inherit_id=False look for mode=primary
    * If root.inherit_id, resolve the arch for root.inherit_id (?using which
      model?), then apply root's inheritance specs to it
    * Apply inheriting views on top
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestViewCombined, self).setUp()
        self.a1 = self.View.create({'model': 'a', 'arch': '<qweb><a1/></qweb>'})
        self.a2 = self.View.create({'model': 'a', 'inherit_id': self.a1.id, 'priority': 5, 'arch': '<xpath expr="//a1" position="after"><a2/></xpath>'})
        self.a3 = self.View.create({'model': 'a', 'inherit_id': self.a1.id, 'arch': '<xpath expr="//a1" position="after"><a3/></xpath>'})
        self.a4 = self.View.create({'model': 'a', 'inherit_id': self.a1.id, 'mode': 'primary', 'arch': '<xpath expr="//a1" position="after"><a4/></xpath>'})
        self.b1 = self.View.create({'model': 'b', 'inherit_id': self.a3.id, 'mode': 'primary', 'arch': '<xpath expr="//a1" position="after"><b1/></xpath>'})
        self.b2 = self.View.create({'model': 'b', 'inherit_id': self.b1.id, 'arch': '<xpath expr="//a1" position="after"><b2/></xpath>'})
        self.c1 = self.View.create({'model': 'c', 'inherit_id': self.a1.id, 'mode': 'primary', 'arch': '<xpath expr="//a1" position="after"><c1/></xpath>'})
        self.c2 = self.View.create({'model': 'c', 'inherit_id': self.c1.id, 'priority': 5, 'arch': '<xpath expr="//a1" position="after"><c2/></xpath>'})
        self.c3 = self.View.create({'model': 'c', 'inherit_id': self.c2.id, 'priority': 10, 'arch': '<xpath expr="//a1" position="after"><c3/></xpath>'})
        self.d1 = self.View.create({'model': 'd', 'inherit_id': self.b1.id, 'mode': 'primary', 'arch': '<xpath expr="//a1" position="after"><d1/></xpath>'})

    def test_basic_read(self):
        if False:
            print('Hello World!')
        context = {'check_view_ids': self.View.search([]).ids}
        arch = self.a1.with_context(context).read_combined(['arch'])['arch']
        self.assertEqual(etree.fromstring(arch), E.qweb(E.a1(), E.a3(), E.a2()), arch)

    def test_read_from_child(self):
        if False:
            print('Hello World!')
        context = {'check_view_ids': self.View.search([]).ids}
        arch = self.a3.with_context(context).read_combined(['arch'])['arch']
        self.assertEqual(etree.fromstring(arch), E.qweb(E.a1(), E.a3(), E.a2()), arch)

    def test_read_from_child_primary(self):
        if False:
            i = 10
            return i + 15
        context = {'check_view_ids': self.View.search([]).ids}
        arch = self.a4.with_context(context).read_combined(['arch'])['arch']
        self.assertEqual(etree.fromstring(arch), E.qweb(E.a1(), E.a4(), E.a3(), E.a2()), arch)

    def test_cross_model_simple(self):
        if False:
            i = 10
            return i + 15
        context = {'check_view_ids': self.View.search([]).ids}
        arch = self.c2.with_context(context).read_combined(['arch'])['arch']
        self.assertEqual(etree.fromstring(arch), E.qweb(E.a1(), E.c3(), E.c2(), E.c1(), E.a3(), E.a2()), arch)

    def test_cross_model_double(self):
        if False:
            while True:
                i = 10
        context = {'check_view_ids': self.View.search([]).ids}
        arch = self.d1.with_context(context).read_combined(['arch'])['arch']
        self.assertEqual(etree.fromstring(arch), E.qweb(E.a1(), E.d1(), E.b2(), E.b1(), E.a3(), E.a2()), arch)

class TestOptionalViews(ViewCase):
    """
    Tests ability to enable/disable inherited views, formerly known as
    inherit_option_id
    """

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestOptionalViews, self).setUp()
        self.v0 = self.View.create({'model': 'a', 'arch': '<qweb><base/></qweb>'})
        self.v1 = self.View.create({'model': 'a', 'inherit_id': self.v0.id, 'active': True, 'priority': 10, 'arch': '<xpath expr="//base" position="after"><v1/></xpath>'})
        self.v2 = self.View.create({'model': 'a', 'inherit_id': self.v0.id, 'active': True, 'priority': 9, 'arch': '<xpath expr="//base" position="after"><v2/></xpath>'})
        self.v3 = self.View.create({'model': 'a', 'inherit_id': self.v0.id, 'active': False, 'priority': 8, 'arch': '<xpath expr="//base" position="after"><v3/></xpath>'})

    def test_applied(self):
        if False:
            i = 10
            return i + 15
        ' mandatory and enabled views should be applied\n        '
        context = {'check_view_ids': self.View.search([]).ids}
        arch = self.v0.with_context(context).read_combined(['arch'])['arch']
        self.assertEqual(etree.fromstring(arch), E.qweb(E.base(), E.v1(), E.v2()))

    def test_applied_state_toggle(self):
        if False:
            while True:
                i = 10
        ' Change active states of v2 and v3, check that the results\n        are as expected\n        '
        self.v2.toggle()
        context = {'check_view_ids': self.View.search([]).ids}
        arch = self.v0.with_context(context).read_combined(['arch'])['arch']
        self.assertEqual(etree.fromstring(arch), E.qweb(E.base(), E.v1()))
        self.v3.toggle()
        context = {'check_view_ids': self.View.search([]).ids}
        arch = self.v0.with_context(context).read_combined(['arch'])['arch']
        self.assertEqual(etree.fromstring(arch), E.qweb(E.base(), E.v1(), E.v3()))
        self.v2.toggle()
        context = {'check_view_ids': self.View.search([]).ids}
        arch = self.v0.with_context(context).read_combined(['arch'])['arch']
        self.assertEqual(etree.fromstring(arch), E.qweb(E.base(), E.v1(), E.v2(), E.v3()))

class TestXPathExtentions(common.BaseCase):

    def test_hasclass(self):
        if False:
            return 10
        tree = E.node(E.node({'class': 'foo bar baz'}), E.node({'class': 'foo bar'}), {'class': 'foo'})
        self.assertEqual(len(tree.xpath('//node[hasclass("foo")]')), 3)
        self.assertEqual(len(tree.xpath('//node[hasclass("bar")]')), 2)
        self.assertEqual(len(tree.xpath('//node[hasclass("baz")]')), 1)
        self.assertEqual(len(tree.xpath('//node[hasclass("foo")][not(hasclass("bar"))]')), 1)
        self.assertEqual(len(tree.xpath('//node[hasclass("foo", "baz")]')), 1)

class TestQWebRender(ViewCase):

    def test_render(self):
        if False:
            return 10
        view1 = self.View.create({'name': 'dummy', 'type': 'qweb', 'arch': '\n                <t t-name="base.dummy">\n                    <div><span>something</span></div>\n                </t>\n        '})
        view2 = self.View.create({'name': 'dummy_ext', 'type': 'qweb', 'inherit_id': view1.id, 'arch': '\n                <xpath expr="//div" position="inside">\n                    <span>another thing</span>\n                </xpath>\n            '})
        view3 = self.View.create({'name': 'dummy_primary_ext', 'type': 'qweb', 'inherit_id': view1.id, 'mode': 'primary', 'arch': '\n                <xpath expr="//div" position="inside">\n                    <span>another primary thing</span>\n                </xpath>\n            '})
        content1 = self.env['ir.qweb'].with_context(check_view_ids=[view1.id, view2.id]).render(view1.id)
        content2 = self.env['ir.qweb'].with_context(check_view_ids=[view1.id, view2.id]).render(view2.id)
        self.assertEqual(content1, content2)
        self.env.cr.execute("INSERT INTO ir_model_data(name, model, res_id, module)VALUES ('dummy', 'ir.ui.view', %s, 'base')" % view1.id)
        self.env.cr.execute("INSERT INTO ir_model_data(name, model, res_id, module)VALUES ('dummy_ext', 'ir.ui.view', %s, 'base')" % view2.id)
        content1 = self.env['ir.qweb'].with_context(check_view_ids=[view1.id, view2.id]).render('base.dummy')
        content2 = self.env['ir.qweb'].with_context(check_view_ids=[view1.id, view2.id]).render('base.dummy_ext')
        self.assertEqual(content1, content2)
        content1 = self.env['ir.qweb'].with_context(check_view_ids=[view1.id, view2.id, view3.id]).render(view1.id)
        content3 = self.env['ir.qweb'].with_context(check_view_ids=[view1.id, view2.id, view3.id]).render(view3.id)
        self.assertNotEqual(content1, content3)
        self.env.cr.execute("INSERT INTO ir_model_data(name, model, res_id, module)VALUES ('dummy_primary_ext', 'ir.ui.view', %s, 'base')" % view3.id)
        content1 = self.env['ir.qweb'].with_context(check_view_ids=[view1.id, view2.id, view3.id]).render('base.dummy')
        content3 = self.env['ir.qweb'].with_context(check_view_ids=[view1.id, view2.id, view3.id]).render('base.dummy_primary_ext')
        self.assertNotEqual(content1, content3)
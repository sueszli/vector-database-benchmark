from odoo.tests import common

class test_inherits(common.TransactionCase):

    def test_00_inherits(self):
        if False:
            for i in range(10):
                print('nop')
        ' Check that a many2one field with delegate=True adds an entry in _inherits '
        daughter = self.env['test.inherit.daughter']
        self.assertEqual(daughter._inherits, {'test.inherit.mother': 'template_id'})

    def test_10_access_from_child_to_parent_model(self):
        if False:
            print('Hello World!')
        ' check whether added field in model is accessible from children models (_inherits) '
        mother = self.env['test.inherit.mother']
        daughter = self.env['test.inherit.daughter']
        self.assertIn('field_in_mother', mother._fields)
        self.assertIn('field_in_mother', daughter._fields)

    def test_20_field_extension(self):
        if False:
            while True:
                i = 10
        ' check the extension of a field in an inherited model '
        mother = self.env['test.inherit.mother']
        daughter = self.env['test.inherit.daughter']
        field = mother._fields['name']
        self.assertTrue(field.required)
        self.assertEqual(field.default(mother), 'Bar')
        self.assertEqual(mother.default_get(['name']), {'name': 'Bar'})
        field = daughter._fields['name']
        self.assertFalse(field.required)
        self.assertEqual(field.default(daughter), 'Baz')
        self.assertEqual(daughter.default_get(['name']), {'name': 'Baz'})
        field = mother._fields['state']
        self.assertFalse(field.default)
        self.assertEqual(mother.default_get(['state']), {})
        field = daughter._fields['template_id']
        self.assertEqual(field.comodel_name, 'test.inherit.mother')
        self.assertEqual(field.string, 'Template')
        self.assertTrue(field.required)

    def test_30_depends_extension(self):
        if False:
            i = 10
            return i + 15
        ' check that @depends on overridden compute methods extends dependencies '
        mother = self.env['test.inherit.mother']
        field = mother._fields['surname']
        self.assertItemsEqual(field.depends, ['name', 'field_in_mother'])

    def test_40_selection_extension(self):
        if False:
            print('Hello World!')
        ' check that attribute selection_add=... extends selection on fields. '
        mother = self.env['test.inherit.mother']
        self.assertEqual(mother._fields['state'].selection, [('a', 'A'), ('b', 'B'), ('c', 'C'), ('d', 'D')])

    def test_50_search_one2many(self):
        if False:
            for i in range(10):
                print('nop')
        ' check search on one2many field based on inherited many2one field. '
        partner_demo = self.env.ref('base.partner_demo')
        daughter = self.env['test.inherit.daughter'].create({'partner_id': partner_demo.id})
        self.assertEqual(daughter.partner_id, partner_demo)
        self.assertIn(daughter, partner_demo.daughter_ids)
        partners = self.env['res.partner'].search([('daughter_ids', 'like', 'not existing daugther')])
        self.assertFalse(partners)
        partners = self.env['res.partner'].search([('daughter_ids', 'not like', 'not existing daugther')])
        self.assertIn(partner_demo, partners)
        partners = self.env['res.partner'].search([('daughter_ids', '!=', False)])
        self.assertIn(partner_demo, partners)
        partners = self.env['res.partner'].search([('daughter_ids', 'in', daughter.ids)])
        self.assertIn(partner_demo, partners)

class test_override_property(common.TransactionCase):

    def test_override_with_normal_field(self):
        if False:
            print('Hello World!')
        ' test overriding a property field by a function field '
        record = self.env['test.inherit.property'].create({'name': 'Stuff'})
        self.assertFalse(record.property_foo)
        self.assertFalse(type(record).property_foo.company_dependent)
        self.assertTrue(type(record).property_foo.store)

    def test_override_with_computed_field(self):
        if False:
            print('Hello World!')
        ' test overriding a property field by a computed field '
        record = self.env['test.inherit.property'].create({'name': 'Stuff'})
        self.assertEqual(record.property_bar, 42)
        self.assertFalse(type(record).property_bar.company_dependent)

class TestInherit(common.TransactionCase):

    def test_extend_parent(self):
        if False:
            print('Hello World!')
        ' test whether a model extension is visible in its children models. '
        parent = self.env['test.inherit.parent']
        child = self.env['test.inherit.child']
        self.assertIn('foo', parent.fields_get())
        self.assertNotIn('bar', parent.fields_get())
        self.assertIn('foo', child.fields_get())
        self.assertIn('bar', child.fields_get())
        self.assertEqual(parent.stuff(), 'P1P2')
        self.assertEqual(child.stuff(), 'P1P2C1')
        self.assertEqual(parent._table, 'test_inherit_parent')
        self.assertEqual(child._table, 'test_inherit_child')
        self.assertEqual(len(parent._sql_constraints), 1)
        self.assertEqual(len(child._sql_constraints), 1)
        self.assertEqual(len(parent._constraint_methods), 1)
        self.assertEqual(len(child._constraint_methods), 1)
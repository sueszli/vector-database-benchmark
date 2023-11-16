from lxml import etree
from odoo.exceptions import AccessError
from odoo.tests.common import TransactionCase
from odoo.tools.misc import mute_logger
USER_DEMO = 'base.user_demo'
GROUP_SYSTEM = 'base.group_system'

class TestACL(TransactionCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestACL, self).setUp()
        self.demo_user = self.env.ref(USER_DEMO)
        self.erp_system_group = self.env.ref(GROUP_SYSTEM)

    def _set_field_groups(self, model, field_name, groups):
        if False:
            for i in range(10):
                print('nop')
        field = model._fields[field_name]
        self.patch(field, 'groups', groups)

    def test_field_visibility_restriction(self):
        if False:
            while True:
                i = 10
        'Check that model-level ``groups`` parameter effectively restricts access to that\n           field for users who do not belong to one of the explicitly allowed groups'
        currency = self.env['res.currency'].sudo(self.demo_user)
        original_fields = currency.fields_get([])
        form_view = currency.fields_view_get(False, 'form')
        view_arch = etree.fromstring(form_view.get('arch'))
        has_group_system = self.demo_user.has_group(GROUP_SYSTEM)
        self.assertFalse(has_group_system, '`demo` user should not belong to the restricted group before the test')
        self.assertIn('decimal_places', original_fields, "'decimal_places' field must be properly visible before the test")
        self.assertNotEquals(view_arch.xpath("//field[@name='decimal_places']"), [], "Field 'decimal_places' must be found in view definition before the test")
        self._set_field_groups(currency, 'decimal_places', GROUP_SYSTEM)
        fields = currency.fields_get([])
        form_view = currency.fields_view_get(False, 'form')
        view_arch = etree.fromstring(form_view.get('arch'))
        self.assertNotIn('decimal_places', fields, "'decimal_places' field should be gone")
        self.assertEquals(view_arch.xpath("//field[@name='decimal_places']"), [], "Field 'decimal_places' must not be found in view definition")
        self.erp_system_group.users += self.demo_user
        has_group_system = self.demo_user.has_group(GROUP_SYSTEM)
        fields = currency.fields_get([])
        form_view = currency.fields_view_get(False, 'form')
        view_arch = etree.fromstring(form_view.get('arch'))
        self.assertTrue(has_group_system, '`demo` user should now belong to the restricted group')
        self.assertIn('decimal_places', fields, "'decimal_places' field must be properly visible again")
        self.assertNotEquals(view_arch.xpath("//field[@name='decimal_places']"), [], "Field 'decimal_places' must be found in view definition again")

    @mute_logger('odoo.models')
    def test_field_crud_restriction(self):
        if False:
            for i in range(10):
                print('nop')
        'Read/Write RPC access to restricted field should be forbidden'
        partner = self.env['res.partner'].browse(1).sudo(self.demo_user)
        has_group_system = self.demo_user.has_group(GROUP_SYSTEM)
        self.assertFalse(has_group_system, '`demo` user should not belong to the restricted group')
        self.assert_(partner.read(['bank_ids']))
        self.assert_(partner.write({'bank_ids': []}))
        self._set_field_groups(partner, 'bank_ids', GROUP_SYSTEM)
        with self.assertRaises(AccessError):
            partner.read(['bank_ids'])
        with self.assertRaises(AccessError):
            partner.write({'bank_ids': []})
        self.erp_system_group.users += self.demo_user
        has_group_system = self.demo_user.has_group(GROUP_SYSTEM)
        self.assertTrue(has_group_system, '`demo` user should now belong to the restricted group')
        self.assert_(partner.read(['bank_ids']))
        self.assert_(partner.write({'bank_ids': []}))

    @mute_logger('odoo.models')
    def test_fields_browse_restriction(self):
        if False:
            return 10
        'Test access to records having restricted fields'
        partner = self.env['res.partner'].sudo(self.demo_user)
        self._set_field_groups(partner, 'email', GROUP_SYSTEM)
        partner = partner.search([], limit=1)
        partner.name
        with self.assertRaises(AccessError):
            with mute_logger('odoo.models'):
                partner.email

    def test_view_create_edit_button_invisibility(self):
        if False:
            i = 10
            return i + 15
        ' Test form view Create, Edit, Delete button visibility based on access right of model'
        methods = ['create', 'edit', 'delete']
        company = self.env['res.company'].sudo(self.demo_user)
        company_view = company.fields_view_get(False, 'form')
        view_arch = etree.fromstring(company_view['arch'])
        for method in methods:
            self.assertEqual(view_arch.get(method), 'false')

    def test_view_create_edit_button_visibility(self):
        if False:
            while True:
                i = 10
        ' Test form view Create, Edit, Delete button visibility based on access right of model'
        self.erp_system_group.users += self.demo_user
        methods = ['create', 'edit', 'delete']
        company = self.env['res.company'].sudo(self.demo_user)
        company_view = company.fields_view_get(False, 'form')
        view_arch = etree.fromstring(company_view['arch'])
        for method in methods:
            self.assertIsNone(view_arch.get(method))

    def test_m2o_field_create_edit_invisibility(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test many2one field Create and Edit option visibility based on access rights of relation field'
        methods = ['create', 'write']
        company = self.env['res.company'].sudo(self.demo_user)
        company_view = company.fields_view_get(False, 'form')
        view_arch = etree.fromstring(company_view['arch'])
        field_node = view_arch.xpath("//field[@name='currency_id']")
        self.assertTrue(len(field_node), 'currency_id field should be in company from view')
        for method in methods:
            self.assertEqual(field_node[0].get('can_' + method), 'false')

    def test_m2o_field_create_edit_visibility(self):
        if False:
            while True:
                i = 10
        ' Test many2one field Create and Edit option visibility based on access rights of relation field'
        self.erp_system_group.users += self.demo_user
        methods = ['create', 'write']
        company = self.env['res.company'].sudo(self.demo_user)
        company_view = company.fields_view_get(False, 'form')
        view_arch = etree.fromstring(company_view['arch'])
        field_node = view_arch.xpath("//field[@name='currency_id']")
        self.assertTrue(len(field_node), 'currency_id field should be in company from view')
        for method in methods:
            self.assertEqual(field_node[0].get('can_' + method), 'true')

class TestIrRule(TransactionCase):

    def test_ir_rule(self):
        if False:
            return 10
        model_res_partner = self.env.ref('base.model_res_partner')
        group_user = self.env.ref('base.group_user')
        user_demo = self.env.ref('base.user_demo')
        rule1 = self.env['ir.rule'].create({'name': 'test_rule1', 'model_id': model_res_partner.id, 'domain_force': False, 'groups': [(6, 0, group_user.ids)]})
        partners_demo = self.env['res.partner'].sudo(user_demo)
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        rule1.domain_force = "[(1,'=',1)]"
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        rule1.domain_force = '[]'
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        rule2 = self.env['ir.rule'].create({'name': 'test_rule2', 'model_id': model_res_partner.id, 'domain_force': False, 'groups': [(6, 0, group_user.ids)]})
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        rule1.domain_force = "[(1,'=',1)]"
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        rule2.domain_force = "[(1,'=',1)]"
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        rule3 = self.env['ir.rule'].create({'name': 'test_rule3', 'model_id': model_res_partner.id, 'domain_force': False, 'groups': [(6, 0, group_user.ids)]})
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        rule3.domain_force = "[(1,'=',1)]"
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        global_rule = self.env.ref('base.res_company_rule_employee')
        global_rule.domain_force = "[('id','child_of',[user.company_id.id])]"
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        rule2.domain_force = "[('id','=',False),('name','=',False)]"
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partner.')
        group_test = self.env['res.groups'].create({'name': 'Test Group', 'users': [(6, 0, user_demo.ids)]})
        rule3.write({'domain_force': "[('name','!=',False),('id','!=',False)]", 'groups': [(6, 0, group_test.ids)]})
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see partners even with the combined rules.')
        self.env['ir.rule'].search([('groups', '=', False)]).unlink()
        partners = partners_demo.search([])
        self.assertTrue(partners, 'Demo user should see some partners.')
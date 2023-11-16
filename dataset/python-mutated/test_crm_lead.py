from .common import TestCrmCases
from odoo.modules.module import get_module_resource

class TestCRMLead(TestCrmCases):

    def test_crm_lead_cancel(self):
        if False:
            for i in range(10):
                print('nop')
        team = self.env['crm.team'].sudo(self.crm_salemanager.id).create({'name': 'Phone Marketing'})
        lead = self.env.ref('crm.crm_case_1')
        lead.sudo(self.crm_salemanager.id).write({'team_id': team.id})
        self.assertEqual(lead.stage_id.sequence, 1, 'Lead is in new stage')

    def test_crm_lead_copy(self):
        if False:
            for i in range(10):
                print('nop')
        self.env.ref('crm.crm_case_4').copy()

    def test_crm_lead_unlink(self):
        if False:
            for i in range(10):
                print('nop')
        self.env.ref('crm.crm_case_4').sudo(self.crm_salemanager.id).unlink()

    def test_find_stage(self):
        if False:
            i = 10
            return i + 15
        lead = self.env['crm.lead'].create({'type': 'lead', 'name': 'Test lead new', 'partner_id': self.env.ref('base.res_partner_1').id, 'description': 'This is the description of the test new lead.', 'team_id': self.env.ref('sales_team.team_sales_department').id})
        lead.convert_opportunity(self.env.ref('base.res_partner_2').id)
        self.assertLessEqual(lead.stage_id.sequence, 1, 'Default stage of lead is incorrect!')
        lead.action_set_won()
        stage_id = lead._stage_find(domain=[('probability', '=', 100.0)])
        self.assertEqual(stage_id, lead.stage_id, 'Stage of opportunity is incorrect!')

    def test_crm_lead_message(self):
        if False:
            print('Hello World!')
        request_file = open(get_module_resource('crm', 'tests', 'customer_request.eml'), 'rb')
        request_message = request_file.read()
        self.env['mail.thread'].sudo(self.crm_salesman).message_process('crm.lead', request_message)
        lead = self.env['crm.lead'].sudo(self.crm_salesman).search([('email_from', '=', 'Mr. John Right <info@customer.com>')], limit=1)
        self.assertTrue(lead.ids, 'Fail to create merge opportunity wizard')
        self.assertFalse(lead.partner_id, 'Customer should be a new one')
        self.assertEqual(lead.name, 'Fournir votre devis avec le meilleur prix.', 'Subject does not match')
        lead = self.env['crm.lead'].search([('email_from', '=', 'Mr. John Right <info@customer.com>')], limit=1)
        mail = self.env['mail.compose.message'].with_context(active_model='crm.lead', active_id=lead.id).create({'body': 'Merci de votre intérêt pour notre produit, nous vous contacterons bientôt. Bien à vous', 'email_from': 'sales@mycompany.com'})
        try:
            mail.send_mail()
        except:
            pass
        lead = self.env['crm.lead'].search([('email_from', '=', 'Mr. John Right <info@customer.com>')], limit=1)
        lead.handle_partner_assignation()

    def test_crm_lead_merge(self):
        if False:
            for i in range(10):
                print('nop')
        default_stage_id = self.ref('crm.stage_lead1')
        LeadSalesmanager = self.env['crm.lead'].sudo(self.crm_salemanager.id)
        test_crm_opp_01 = LeadSalesmanager.create({'type': 'opportunity', 'name': 'Test opportunity 1', 'partner_id': self.env.ref('base.res_partner_3').id, 'stage_id': default_stage_id, 'description': 'This is the description of the test opp 1.'})
        test_crm_lead_01 = LeadSalesmanager.create({'type': 'lead', 'name': 'Test lead first', 'partner_id': self.env.ref('base.res_partner_1').id, 'stage_id': default_stage_id, 'description': 'This is the description of the test lead first.'})
        test_crm_lead_02 = LeadSalesmanager.create({'type': 'lead', 'name': 'Test lead second', 'partner_id': self.env.ref('base.res_partner_1').id, 'stage_id': default_stage_id, 'description': 'This is the description of the test lead second.'})
        lead_ids = [test_crm_opp_01.id, test_crm_lead_01.id, test_crm_lead_02.id]
        additionnal_context = {'active_model': 'crm.lead', 'active_ids': lead_ids, 'active_id': lead_ids[0]}
        merge_opp_wizard_01 = self.env['crm.merge.opportunity'].sudo(self.crm_salemanager.id).with_context(**additionnal_context).create({})
        merge_opp_wizard_01.action_merge()
        merged_lead = self.env['crm.lead'].search([('name', '=', 'Test opportunity 1'), ('partner_id', '=', self.env.ref('base.res_partner_3').id)], limit=1)
        self.assertTrue(merged_lead, 'Fail to create merge opportunity wizard')
        self.assertEqual(merged_lead.description, 'This is the description of the test opp 1.\n\nThis is the description of the test lead first.\n\nThis is the description of the test lead second.', 'Description mismatch: when merging leads/opps with different text values, these values should get concatenated and separated with line returns')
        self.assertEqual(merged_lead.type, 'opportunity', 'Type mismatch: when at least one opp in involved in the merge, the result should be a new opp (instead of %s)' % merged_lead.type)
        self.assertFalse(test_crm_lead_01.exists(), 'This tailing lead (id %s) should not exist anymore' % test_crm_lead_02.id)
        self.assertFalse(test_crm_lead_02.exists(), 'This tailing opp (id %s) should not exist anymore' % test_crm_opp_01.id)
        test_crm_lead_03 = LeadSalesmanager.create({'type': 'lead', 'name': 'Test lead 3', 'partner_id': self.env.ref('base.res_partner_1').id, 'stage_id': default_stage_id})
        test_crm_lead_04 = LeadSalesmanager.create({'type': 'lead', 'name': 'Test lead 4', 'partner_id': self.env.ref('base.res_partner_1').id, 'stage_id': default_stage_id})
        lead_ids = [test_crm_lead_03.id, test_crm_lead_04.id]
        additionnal_context = {'active_model': 'crm.lead', 'active_ids': lead_ids, 'active_id': lead_ids[0]}
        merge_opp_wizard_02 = self.env['crm.merge.opportunity'].sudo(self.crm_salemanager.id).with_context(**additionnal_context).create({})
        merge_opp_wizard_02.action_merge()
        merged_lead = self.env['crm.lead'].search([('name', '=', 'Test lead 3'), ('partner_id', '=', self.env.ref('base.res_partner_1').id)], limit=1)
        self.assertTrue(merged_lead, 'Fail to create merge opportunity wizard')
        self.assertEqual(merged_lead.partner_id.id, self.env.ref('base.res_partner_1').id, 'Partner mismatch')
        self.assertEqual(merged_lead.type, 'lead', 'Type mismatch: when leads get merged together, the result should be a new lead (instead of %s)' % merged_lead.type)
        self.assertFalse(test_crm_lead_04.exists(), 'This tailing lead (id %s) should not exist anymore' % test_crm_lead_04.id)
        test_crm_opp_02 = LeadSalesmanager.create({'type': 'opportunity', 'name': 'Test opportunity 2', 'partner_id': self.env.ref('base.res_partner_3').id, 'stage_id': default_stage_id})
        test_crm_opp_03 = LeadSalesmanager.create({'type': 'opportunity', 'name': 'Test opportunity 3', 'partner_id': self.env.ref('base.res_partner_3').id, 'stage_id': default_stage_id})
        opportunity_ids = [test_crm_opp_02.id, test_crm_opp_03.id]
        additionnal_context = {'active_model': 'crm.lead', 'active_ids': opportunity_ids, 'active_id': opportunity_ids[0]}
        merge_opp_wizard_03 = self.env['crm.merge.opportunity'].sudo(self.crm_salemanager.id).with_context(**additionnal_context).create({})
        merge_opp_wizard_03.action_merge()
        merged_opportunity = self.env['crm.lead'].search([('name', '=', 'Test opportunity 2'), ('partner_id', '=', self.env.ref('base.res_partner_3').id)], limit=1)
        self.assertTrue(merged_opportunity, 'Fail to create merge opportunity wizard')
        self.assertEqual(merged_opportunity.partner_id.id, self.env.ref('base.res_partner_3').id, 'Partner mismatch')
        self.assertEqual(merged_opportunity.type, 'opportunity', 'Type mismatch: when opps get merged together, the result should be a new opp (instead of %s)' % merged_opportunity.type)
        self.assertFalse(test_crm_opp_03.exists(), 'This tailing opp (id %s) should not exist anymore' % test_crm_opp_03.id)
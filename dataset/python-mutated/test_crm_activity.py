from .common import TestCrmCases
from odoo import fields
from datetime import date

class TestCrmActivity(TestCrmCases):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestCrmActivity, self).setUp()
        Activity = self.env['crm.activity']
        self.activity3 = Activity.create({'name': 'Celebrate the sale', 'days': 3, 'description': 'ACT 3 : Beers for everyone because I am a good salesman !', 'internal': True, 'res_model': 'crm.lead'})
        self.activity2 = Activity.create({'name': 'Call for Demo', 'days': 6, 'description': 'ACT 2 : I want to show you my ERP !', 'internal': True, 'res_model': 'crm.lead', 'recommended_activity_ids': [(6, 0, [self.activity3.id])]})
        self.activity1 = Activity.create({'name': 'Initial Contact', 'days': 5, 'description': 'ACT 1 : Presentation, barbecue, ... ', 'internal': True, 'res_model': 'crm.lead', 'recommended_activity_ids': [(6, 0, [self.activity2.id])]})
        self.partner_client = self.env.ref('base.res_partner_1')
        Lead = self.env['crm.lead'].sudo(self.crm_salesman.id)
        self.lead = Lead.create({'type': 'opportunity', 'name': 'Test Opportunity Activity Log', 'partner_id': self.partner_client.id, 'team_id': self.env.ref('sales_team.team_sales_department').id, 'user_id': self.crm_salesman.id})

    def test_crm_activity_recipients(self):
        if False:
            return 10
        ' This test case check :\n                - no internal subtype followed by client\n                - activity subtype are not default ones\n                - only activity followers are recipients when this kind of activity is logged\n        '
        activity = self.activity2
        self.lead.message_subscribe([self.partner_client.id])
        is_internal_subtype_for_client = self.lead.message_follower_ids.filtered(lambda fol: fol.partner_id.id == self.partner_client.id).mapped('subtype_ids.internal')
        self.assertFalse(any(is_internal_subtype_for_client), 'Partner client is following an internal subtype')
        self.lead.message_subscribe([self.crm_salemanager.partner_id.id])
        manager_follower = self.env['mail.followers'].sudo().search([('res_model', '=', 'crm.lead'), ('res_id', '=', self.lead.id), ('partner_id', '=', self.crm_salemanager.partner_id.id)])
        manager_follower.write({'subtype_ids': [(4, activity.subtype_id.id)]})
        ActivityLogWizard = self.env['crm.activity.log'].sudo(self.crm_salesman.id)
        wizard = ActivityLogWizard.create({'note': 'Content of the activity to log', 'lead_id': self.lead.id})
        wizard.onchange_lead_id()
        wizard.write({'next_activity_id': activity.id})
        wizard.action_log()
        activity_message = self.lead.message_ids[0]
        self.assertEqual(activity_message.needaction_partner_ids, self.crm_salemanager.partner_id, 'Only the crm manager should be notified by the activity')
        self.assertEqual(self.lead.next_activity_id.id, False, 'When logging activity, the next activity planned is erased')

    def test_crm_activity_next_action(self):
        if False:
            i = 10
            return i + 15
        ' This test case set the next activity on a lead, log another, and schedule a third. '
        self.lead.write({'next_activity_id': self.activity1.id})
        self.lead._onchange_next_activity_id()
        self.assertEqual(self.lead.title_action, self.activity1.description, 'Activity title should be the same on the lead and on the chosen activity')
        wizard = self.env['crm.activity.log'].sudo(self.crm_salesman.id).create({'note': 'Content of the activity to log', 'lead_id': self.lead.id})
        wizard.onchange_lead_id()
        wizard.write({'next_activity_id': self.activity2.id})
        wizard.action_log()
        self.assertFalse(self.lead.next_activity_id.id, 'No next activity should be set on lead, since we jsut log another activity')
        self.env['crm.activity.log'].sudo(self.crm_salesman.id).create({'next_activity_id': self.activity3.id, 'note': 'Content of the activity to log', 'lead_id': self.lead.id})
        wizard.onchange_lead_id()
        wizard.write({'next_activity_id': self.activity3.id})
        wizard.onchange_next_activity_id()
        wizard.action_schedule()
        delta_days = (fields.Date.from_string(self.lead.date_action) - date.today()).days
        self.assertEqual(self.activity3.days, delta_days, 'The action date should be in the number of days set up on the activity 3')
        self.assertEqual(self.lead.title_action, self.activity3.description, 'Activity title should be the same on the lead and on the activity 3')
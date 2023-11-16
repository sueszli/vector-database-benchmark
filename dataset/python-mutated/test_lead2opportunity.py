from .common import TestCrmCases

class TestLead2opportunity2win(TestCrmCases):

    def test_lead2opportunity2win(self):
        if False:
            for i in range(10):
                print('nop')
        ' Tests for Test Lead 2 opportunity 2 win '
        CrmLead2OpportunityPartnerMass = self.env['crm.lead2opportunity.partner.mass']
        CalendarAttendee = self.env['calendar.attendee']
        default_stage_id = self.ref('crm.stage_lead1')
        crm_case_2 = self.env.ref('crm.crm_case_2')
        crm_case_3 = self.env.ref('crm.crm_case_3')
        crm_case_13 = self.env.ref('crm.crm_case_13')
        crm_case_3.write({'stage_id': default_stage_id})
        self.assertEqual(crm_case_3.stage_id.sequence, 1, 'Lead stage is Open')
        crm_case_3.sudo(self.crm_salemanager.id).convert_opportunity(self.env.ref('base.res_partner_2').id)
        self.assertEqual(crm_case_3.type, 'opportunity', 'Lead is not converted to opportunity!')
        self.assertEqual(crm_case_3.partner_id.id, self.env.ref('base.res_partner_2').id, 'Partner mismatch!')
        self.assertEqual(crm_case_3.stage_id.id, default_stage_id, 'Stage of opportunity is incorrect!')
        crm_case_3.action_schedule_meeting()
        crm_case_3.message_post(subject='Test note', body='Détails envoyés par le client sur \u200b\u200ble FAX pour la qualité')
        mass = CrmLead2OpportunityPartnerMass.with_context({'active_model': 'crm.lead', 'active_ids': [crm_case_13.id, crm_case_2.id], 'active_id': crm_case_13.id}).create({'user_ids': [(6, 0, self.env.ref('base.user_root').ids)], 'team_id': self.env.ref('sales_team.team_sales_department').id})
        mass.sudo(self.crm_salemanager.id).mass_convert()
        self.assertEqual(crm_case_13.name, 'Plan to buy 60 keyboards and mouses', 'Opportunity name not correct')
        self.assertEqual(crm_case_13.type, 'opportunity', 'Lead is not converted to opportunity!')
        expected_partner = 'Will McEncroe'
        self.assertEqual(crm_case_13.partner_id.name, expected_partner, 'Partner mismatch! %s vs %s' % (crm_case_13.partner_id.name, expected_partner))
        self.assertEqual(crm_case_13.stage_id.id, default_stage_id, 'Stage of probability is incorrect!')
        self.assertEqual(crm_case_2.name, 'Interest in Your New Software', 'Opportunity name not correct')
        self.assertEqual(crm_case_2.type, 'opportunity', 'Lead is not converted to opportunity!')
        self.assertEqual(crm_case_2.stage_id.id, default_stage_id, 'Stage of probability is incorrect!')
        crm_case_2.action_set_lost()
        self.assertEqual(crm_case_2.probability, 0.0, 'Revenue probability should be 0.0!')
        self.env.ref('calendar.calendar_event_4').with_context({'active_model': 'calendar.event'}).write({'state': 'open'})
        CalendarAttendee.create({'partner_id': self.ref('base.partner_root'), 'email': 'user@meeting.com'}).do_accept()

    def test_lead2opportunity_assign_salesmen(self):
        if False:
            i = 10
            return i + 15
        ' Tests for Test Lead2opportunity Assign Salesmen '
        CrmLead2OpportunityPartnerMass = self.env['crm.lead2opportunity.partner.mass']
        LeadSaleman = self.env['crm.lead'].sudo(self.crm_salesman.id)
        default_stage_id = self.ref('crm.stage_lead1')
        test_res_user_01 = self.env['res.users'].create({'name': 'Test user A', 'login': 'tua@example.com', 'new_password': 'tua'})
        test_res_user_02 = self.env['res.users'].create({'name': 'Test user B', 'login': 'tub@example.com', 'new_password': 'tub'})
        test_res_user_03 = self.env['res.users'].create({'name': 'Test user C', 'login': 'tuc@example.com', 'new_password': 'tuc'})
        test_res_user_04 = self.env['res.users'].create({'name': 'Test user D', 'login': 'tud@example.com', 'new_password': 'tud'})
        test_crm_lead_01 = LeadSaleman.create({'type': 'lead', 'name': 'Test lead 1', 'email_from': 'Raoul Grosbedon <raoul@grosbedon.fr>', 'stage_id': default_stage_id})
        test_crm_lead_02 = LeadSaleman.create({'type': 'lead', 'name': 'Test lead 2', 'email_from': 'Raoul Grosbedon <raoul@grosbedon.fr>', 'stage_id': default_stage_id})
        test_crm_lead_03 = LeadSaleman.create({'type': 'lead', 'name': 'Test lead 3', 'email_from': 'Raoul Grosbedon <raoul@grosbedon.fr>', 'stage_id': default_stage_id})
        test_crm_lead_04 = LeadSaleman.create({'type': 'lead', 'name': 'Test lead 4', 'email_from': 'Fabrice Lepoilu', 'stage_id': default_stage_id})
        test_crm_lead_05 = LeadSaleman.create({'type': 'lead', 'name': 'Test lead 5', 'email_from': 'Fabrice Lepoilu', 'stage_id': default_stage_id})
        test_crm_lead_06 = LeadSaleman.create({'type': 'lead', 'name': 'Test lead 6', 'email_from': 'Agrolait SuperSeed SA', 'stage_id': default_stage_id})
        lead_ids = [test_crm_lead_01.id, test_crm_lead_02.id, test_crm_lead_03.id, test_crm_lead_04.id, test_crm_lead_05.id, test_crm_lead_06.id]
        salesmen_ids = [test_res_user_01.id, test_res_user_02.id, test_res_user_03.id, test_res_user_04.id]
        additionnal_context = {'active_model': 'crm.lead', 'active_ids': lead_ids, 'active_id': test_crm_lead_01.id}
        mass = CrmLead2OpportunityPartnerMass.sudo(self.crm_salesman.id).with_context(**additionnal_context).create({'user_ids': [(6, 0, salesmen_ids)], 'team_id': self.env.ref('sales_team.team_sales_department').id, 'deduplicate': False, 'force_assignation': True})
        mass.sudo(self.crm_salesman.id).mass_convert()
        opps = self.env['crm.lead'].sudo(self.crm_salesman.id).browse(lead_ids)
        i = 0
        for opp in opps:
            self.assertEqual(opp.type, 'opportunity', 'Type mismatch: this should be an opp, not a lead')
            self.assertEqual(opp.user_id.id, salesmen_ids[i], 'Salesman mismatch: expected salesman %r, got %r' % (salesmen_ids[i], opp.user_id.id))
            i = i + 1 if i < len(salesmen_ids) - 1 else 0
from odoo.tests.common import TransactionCase

class TestPartnerAssign(TransactionCase):

    def test_00_partner_assign(self):
        if False:
            i = 10
            return i + 15
        partner2 = self.env.ref('base.res_partner_2')
        lead = self.env.ref('crm.crm_case_21')
        '\n            In order to test find nearest Partner functionality and assign to opportunity,\n            I Set Geo Lattitude and Longitude according to partner address.\n        '
        partner2.geo_localize()
        self.assertTrue(50 < partner2.partner_latitude < 51, 'Latitude is wrong: 50 < %s < 51' % partner2.partner_latitude)
        self.assertTrue(3 < partner2.partner_longitude < 5, 'Longitude is wrong: 3 < %s < 5' % partner2.partner_longitude)
        lead.assign_partner()
        self.assertEqual(lead.partner_assigned_id, self.env.ref('base.res_partner_18'), 'Opportuniy is not assigned nearest partner')
        self.assertTrue(50 < lead.partner_latitude < 55, 'Latitude is wrong: 50 < %s < 55' % lead.partner_latitude)
        self.assertTrue(-4 < lead.partner_longitude < -1, 'Longitude is wrong: -4 < %s < -1' % lead.partner_longitude)
        context = dict(self.env.context, default_model='crm.lead', default_res_id=lead.id, active_ids=lead.ids)
        lead_forwarded = self.env['crm.lead.forward.to.partner'].with_context(context).create({})
        try:
            lead_forwarded.action_forward()
        except:
            pass
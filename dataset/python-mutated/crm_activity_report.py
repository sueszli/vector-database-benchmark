from odoo import fields, models, tools

class ActivityReport(models.Model):
    """ CRM Lead Analysis """
    _name = 'crm.activity.report'
    _auto = False
    _description = 'CRM Activity Analysis'
    _rec_name = 'id'
    date = fields.Datetime('Date', readonly=True)
    author_id = fields.Many2one('res.partner', 'Created By', readonly=True)
    user_id = fields.Many2one('res.users', 'Salesperson', readonly=True)
    team_id = fields.Many2one('crm.team', 'Sales Team', readonly=True)
    lead_id = fields.Many2one('crm.lead', 'Lead', readonly=True)
    subject = fields.Char('Summary', readonly=True)
    subtype_id = fields.Many2one('mail.message.subtype', 'Activity', readonly=True)
    country_id = fields.Many2one('res.country', 'Country', readonly=True)
    company_id = fields.Many2one('res.company', 'Company', readonly=True)
    stage_id = fields.Many2one('crm.stage', 'Stage', readonly=True)
    partner_id = fields.Many2one('res.partner', 'Partner/Customer', readonly=True)
    lead_type = fields.Char(string='Type', selection=[('lead', 'Lead'), ('opportunity', 'Opportunity')], help='Type is used to separate Leads and Opportunities')
    active = fields.Boolean('Active', readonly=True)
    probability = fields.Float('Probability', group_operator='avg', readonly=True)

    def init(self):
        if False:
            i = 10
            return i + 15
        tools.drop_view_if_exists(self._cr, 'crm_activity_report')
        self._cr.execute('\n            CREATE VIEW crm_activity_report AS (\n                select\n                    m.id,\n                    m.subtype_id,\n                    m.author_id,\n                    m.date,\n                    m.subject,\n                    l.id as lead_id,\n                    l.user_id,\n                    l.team_id,\n                    l.country_id,\n                    l.company_id,\n                    l.stage_id,\n                    l.partner_id,\n                    l.type as lead_type,\n                    l.active,\n                    l.probability\n                from\n                    "mail_message" m\n                join\n                    "crm_lead" l\n                on\n                    (m.res_id = l.id)\n                WHERE\n                    (m.model = \'crm.lead\')\n            )')
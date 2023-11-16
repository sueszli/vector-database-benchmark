from odoo import fields, models, tools
from ..models import crm_stage

class OpportunityReport(models.Model):
    """ CRM Opportunity Analysis """
    _name = 'crm.opportunity.report'
    _auto = False
    _description = 'CRM Opportunity Analysis'
    _rec_name = 'date_deadline'
    date_deadline = fields.Date('Expected Closing', readonly=True)
    create_date = fields.Datetime('Creation Date', readonly=True)
    opening_date = fields.Datetime('Assignation Date', readonly=True)
    date_closed = fields.Datetime('Close Date', readonly=True)
    date_last_stage_update = fields.Datetime('Last Stage Update', readonly=True)
    active = fields.Boolean('Active', readonly=True)
    delay_open = fields.Float('Delay to Assign', digits=(16, 2), readonly=True, group_operator='avg', help='Number of Days to open the case')
    delay_close = fields.Float('Delay to Close', digits=(16, 2), readonly=True, group_operator='avg', help='Number of Days to close the case')
    delay_expected = fields.Float('Overpassed Deadline', digits=(16, 2), readonly=True, group_operator='avg')
    user_id = fields.Many2one('res.users', string='User', readonly=True)
    team_id = fields.Many2one('crm.team', 'Sales Team', oldname='section_id', readonly=True)
    nbr_activities = fields.Integer('# of Activities', readonly=True)
    city = fields.Char('City')
    country_id = fields.Many2one('res.country', string='Country', readonly=True)
    probability = fields.Float(string='Probability', digits=(16, 2), readonly=True, group_operator='avg')
    total_revenue = fields.Float(string='Total Revenue', digits=(16, 2), readonly=True)
    expected_revenue = fields.Float(string='Probable Turnover', digits=(16, 2), readonly=True)
    stage_id = fields.Many2one('crm.stage', string='Stage', readonly=True, domain="['|', ('team_id', '=', False), ('team_id', '=', team_id)]")
    stage_name = fields.Char(string='Stage Name', readonly=True)
    partner_id = fields.Many2one('res.partner', string='Partner', readonly=True)
    company_id = fields.Many2one('res.company', string='Company', readonly=True)
    priority = fields.Selection(crm_stage.AVAILABLE_PRIORITIES, string='Priority', group_operator='avg')
    type = fields.Selection([('lead', 'Lead'), ('opportunity', 'Opportunity')], help='Type is used to separate Leads and Opportunities')
    lost_reason = fields.Many2one('crm.lost.reason', string='Lost Reason', readonly=True)
    date_conversion = fields.Datetime(string='Conversion Date', readonly=True)
    campaign_id = fields.Many2one('utm.campaign', string='Campaign', readonly=True)
    source_id = fields.Many2one('utm.source', string='Source', readonly=True)
    medium_id = fields.Many2one('utm.medium', string='Medium', readonly=True)

    def init(self):
        if False:
            i = 10
            return i + 15
        tools.drop_view_if_exists(self._cr, 'crm_opportunity_report')
        self._cr.execute('\n            CREATE VIEW crm_opportunity_report AS (\n                SELECT\n                    c.id,\n                    c.date_deadline,\n\n                    c.date_open as opening_date,\n                    c.date_closed as date_closed,\n                    c.date_last_stage_update as date_last_stage_update,\n\n                    c.user_id,\n                    c.probability,\n                    c.stage_id,\n                    stage.name as stage_name,\n                    c.type,\n                    c.company_id,\n                    c.priority,\n                    c.team_id,\n                    (SELECT COUNT(*)\n                     FROM mail_message m\n                     WHERE m.model = \'crm.lead\' and m.res_id = c.id) as nbr_activities,\n                    c.active,\n                    c.campaign_id,\n                    c.source_id,\n                    c.medium_id,\n                    c.partner_id,\n                    c.city,\n                    c.country_id,\n                    c.planned_revenue as total_revenue,\n                    c.planned_revenue*(c.probability/100) as expected_revenue,\n                    c.create_date as create_date,\n                    extract(\'epoch\' from (c.date_closed-c.create_date))/(3600*24) as  delay_close,\n                    abs(extract(\'epoch\' from (c.date_deadline - c.date_closed))/(3600*24)) as  delay_expected,\n                    extract(\'epoch\' from (c.date_open-c.create_date))/(3600*24) as  delay_open,\n                    c.lost_reason,\n                    c.date_conversion as date_conversion\n                FROM\n                    "crm_lead" c\n                LEFT JOIN "crm_stage" stage\n                ON stage.id = c.stage_id\n                GROUP BY c.id, stage.name\n            )')
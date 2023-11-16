from odoo import api, fields, models
from odoo import tools
from odoo.addons.crm.models import crm_stage

class CrmLeadReportAssign(models.Model):
    """ CRM Lead Report """
    _name = 'crm.lead.report.assign'
    _auto = False
    _description = 'CRM Lead Report'
    partner_assigned_id = fields.Many2one('res.partner', 'Partner', readonly=True)
    grade_id = fields.Many2one('res.partner.grade', 'Grade', readonly=True)
    user_id = fields.Many2one('res.users', 'User', readonly=True)
    country_id = fields.Many2one('res.country', 'Country', readonly=True)
    team_id = fields.Many2one('crm.team', 'Sales Team', oldname='section_id', readonly=True)
    company_id = fields.Many2one('res.company', 'Company', readonly=True)
    date_assign = fields.Date('Assign Date', readonly=True)
    create_date = fields.Datetime('Create Date', readonly=True)
    delay_open = fields.Float('Delay to Assign', digits=(16, 2), readonly=True, group_operator='avg', help='Number of Days to open the case')
    delay_close = fields.Float('Delay to Close', digits=(16, 2), readonly=True, group_operator='avg', help='Number of Days to close the case')
    delay_expected = fields.Float('Overpassed Deadline', digits=(16, 2), readonly=True, group_operator='avg')
    probability = fields.Float('Avg Probability', digits=(16, 2), readonly=True, group_operator='avg')
    probability_max = fields.Float('Max Probability', digits=(16, 2), readonly=True, group_operator='max')
    planned_revenue = fields.Float('Planned Revenue', digits=(16, 2), readonly=True)
    probable_revenue = fields.Float('Probable Revenue', digits=(16, 2), readonly=True)
    tag_ids = fields.Many2many('crm.lead.tag', 'crm_lead_tag_rel', 'lead_id', 'tag_id', 'Tags')
    partner_id = fields.Many2one('res.partner', 'Customer', readonly=True)
    opening_date = fields.Datetime('Opening Date', readonly=True)
    date_closed = fields.Datetime('Close Date', readonly=True)
    nbr_cases = fields.Integer('# of Cases', readonly=True, oldname='nbr')
    company_id = fields.Many2one('res.company', 'Company', readonly=True)
    priority = fields.Selection(crm_stage.AVAILABLE_PRIORITIES, 'Priority')
    type = fields.Selection([('lead', 'Lead'), ('opportunity', 'Opportunity')], 'Type', help='Type is used to separate Leads and Opportunities')

    @api.model_cr
    def init(self):
        if False:
            print('Hello World!')
        '\n            CRM Lead Report\n            @param cr: the current row, from the database cursor\n        '
        tools.drop_view_if_exists(self._cr, 'crm_lead_report_assign')
        self._cr.execute("\n            CREATE OR REPLACE VIEW crm_lead_report_assign AS (\n                SELECT\n                    c.id,\n                    c.date_open as opening_date,\n                    c.date_closed as date_closed,\n                    c.date_assign,\n                    c.user_id,\n                    c.probability,\n                    c.probability as probability_max,\n                    c.type,\n                    c.company_id,\n                    c.priority,\n                    c.team_id,\n                    c.partner_id,\n                    c.country_id,\n                    c.planned_revenue,\n                    c.partner_assigned_id,\n                    p.grade_id,\n                    p.date as partner_date,\n                    c.planned_revenue*(c.probability/100) as probable_revenue,\n                    1 as nbr,\n                    c.create_date as create_date,\n                    extract('epoch' from (c.write_date-c.create_date))/(3600*24) as  delay_close,\n                    extract('epoch' from (c.date_deadline - c.date_closed))/(3600*24) as  delay_expected,\n                    extract('epoch' from (c.date_open-c.create_date))/(3600*24) as  delay_open\n                FROM\n                    crm_lead c\n                    left join res_partner p on (c.partner_assigned_id=p.id)\n            )")
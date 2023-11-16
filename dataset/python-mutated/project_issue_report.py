from odoo import api, fields, models, tools

class ProjectIssueReport(models.Model):
    _name = 'project.issue.report'
    _auto = False
    company_id = fields.Many2one('res.company', 'Company', readonly=True)
    opening_date = fields.Datetime('Date of Opening', readonly=True)
    create_date = fields.Datetime('Create Date', readonly=True)
    date_closed = fields.Datetime('Date of Closing', readonly=True)
    date_last_stage_update = fields.Datetime('Last Stage Update', readonly=True)
    stage_id = fields.Many2one('project.task.type', 'Stage')
    nbr_issues = fields.Integer('# of Issues', readonly=True)
    working_hours_open = fields.Float('Avg. Working Hours to Open', readonly=True, group_operator='avg')
    working_hours_close = fields.Float('Avg. Working Hours to Close', readonly=True, group_operator='avg')
    delay_open = fields.Float('Avg. Delay to Open', digits=(16, 2), readonly=True, group_operator='avg', help='Number of Days to open the project issue.')
    delay_close = fields.Float('Avg. Delay to Close', digits=(16, 2), readonly=True, group_operator='avg', help='Number of Days to close the project issue')
    priority = fields.Selection([('0', 'Low'), ('1', 'Normal'), ('2', 'High')])
    project_id = fields.Many2one('project.project', 'Project', readonly=True)
    user_id = fields.Many2one('res.users', 'Assigned to', readonly=True)
    partner_id = fields.Many2one('res.partner', 'Contact')
    email = fields.Integer('# Emails', readonly=True)

    @api.model_cr
    def init(self):
        if False:
            i = 10
            return i + 15
        tools.drop_view_if_exists(self._cr, 'project_issue_report')
        self._cr.execute("\n            CREATE OR REPLACE VIEW project_issue_report AS (\n                SELECT\n                    c.id as id,\n                    c.date_open as opening_date,\n                    c.create_date as create_date,\n                    c.date_last_stage_update as date_last_stage_update,\n                    c.user_id,\n                    c.working_hours_open,\n                    c.working_hours_close,\n                    c.stage_id,\n                    c.date_closed as date_closed,\n                    c.company_id as company_id,\n                    c.priority as priority,\n                    c.project_id as project_id,\n                    1 as nbr_issues,\n                    c.partner_id,\n                    c.day_open as delay_open,\n                    c.day_close as delay_close,\n                    (SELECT count(id) FROM mail_message WHERE model='project.issue' AND message_type IN ('email', 'comment') AND res_id=c.id) AS email\n\n                FROM\n                    project_issue c\n                LEFT JOIN project_task t on c.task_id = t.id\n                WHERE c.active= 'true'\n            )")
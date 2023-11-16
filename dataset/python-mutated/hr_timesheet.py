from odoo import api, fields, models

class AccountAnalyticLine(models.Model):
    _inherit = 'account.analytic.line'
    task_id = fields.Many2one('project.task', 'Task')
    project_id = fields.Many2one('project.project', 'Project', domain=[('allow_timesheets', '=', True)])
    department_id = fields.Many2one('hr.department', 'Department', related='user_id.employee_ids.department_id', store=True, readonly=True)

    @api.onchange('project_id')
    def onchange_project_id(self):
        if False:
            while True:
                i = 10
        self.task_id = False

    @api.model
    def create(self, vals):
        if False:
            for i in range(10):
                print('nop')
        if vals.get('project_id'):
            project = self.env['project.project'].browse(vals.get('project_id'))
            vals['account_id'] = project.analytic_account_id.id
        return super(AccountAnalyticLine, self).create(vals)

    @api.multi
    def write(self, vals):
        if False:
            i = 10
            return i + 15
        if vals.get('project_id'):
            project = self.env['project.project'].browse(vals.get('project_id'))
            vals['account_id'] = project.analytic_account_id.id
        return super(AccountAnalyticLine, self).write(vals)
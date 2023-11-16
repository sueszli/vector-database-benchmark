from odoo import fields, models

class ReportProjectTaskUser(models.Model):
    _inherit = 'report.project.task.user'
    hours_planned = fields.Float('Planned Hours', readonly=True)
    hours_effective = fields.Float('Effective Hours', readonly=True)
    hours_delay = fields.Float('Avg. Plan.-Eff.', readonly=True)
    remaining_hours = fields.Float('Remaining Hours', readonly=True)
    progress = fields.Float('Progress', group_operator='avg', readonly=True)
    total_hours = fields.Float('Total Hours', readonly=True)

    def _select(self):
        if False:
            while True:
                i = 10
        return super(ReportProjectTaskUser, self)._select() + ',\n            progress as progress,\n            t.effective_hours as hours_effective,\n            remaining_hours as remaining_hours,\n            total_hours as total_hours,\n            t.delay_hours as hours_delay,\n            planned_hours as hours_planned'

    def _group_by(self):
        if False:
            for i in range(10):
                print('nop')
        return super(ReportProjectTaskUser, self)._group_by() + ',\n            remaining_hours,\n            t.effective_hours,\n            progress,\n            total_hours,\n            planned_hours,\n            hours_delay'
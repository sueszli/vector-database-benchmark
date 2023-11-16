from odoo import models

class ProjectStage(models.Model):
    _inherit = ['project.task.type']

    def _get_mail_template_id_domain(self):
        if False:
            return 10
        domain = super(ProjectStage, self)._get_mail_template_id_domain()
        return ['|'] + domain + [('model', '=', 'project.issue')]
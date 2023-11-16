from odoo import api, models, _

class BaseModuleConfiguration(models.TransientModel):
    _name = 'base.module.configuration'

    @api.multi
    def start(self):
        if False:
            return 10
        todos_domain = ['|', ('type', '=', 'recurring'), ('state', '=', 'open')]
        if self.env['ir.actions.todo'].search_count(todos_domain):
            return self.env['res.config'].start()
        else:
            view = self.env.ref('base.view_base_module_configuration_form')
            return {'name': _('System Configuration done'), 'view_type': 'form', 'view_mode': 'form', 'res_model': 'base.mdule.configuration', 'view_id': [view.id], 'type': 'ir.actions.act_window', 'target': 'new'}
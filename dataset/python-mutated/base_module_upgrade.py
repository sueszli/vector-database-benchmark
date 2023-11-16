import odoo
from odoo import api, fields, models, _
from odoo.exceptions import UserError

class BaseModuleUpgrade(models.TransientModel):
    _name = 'base.module.upgrade'
    _description = 'Module Upgrade'

    @api.model
    @api.returns('ir.module.module')
    def get_module_list(self):
        if False:
            i = 10
            return i + 15
        states = ['to upgrade', 'to remove', 'to install']
        return self.env['ir.module.module'].search([('state', 'in', states)])

    @api.model
    def _default_module_info(self):
        if False:
            while True:
                i = 10
        return '\n'.join(('%s: %s' % (mod.name, mod.state) for mod in self.get_module_list()))
    module_info = fields.Text('Apps to Update', readonly=True, default=_default_module_info)

    @api.model
    def fields_view_get(self, view_id=None, view_type='form', toolbar=False, submenu=False):
        if False:
            print('Hello World!')
        res = super(BaseModuleUpgrade, self).fields_view_get(view_id, view_type, toolbar=toolbar, submenu=False)
        if view_type != 'form':
            return res
        if not (self._context.get('active_model') and self._context.get('active_id')):
            return res
        if not self.get_module_list():
            res['arch'] = '<form string="Upgrade Completed" version="7.0">\n                                <separator string="Upgrade Completed" colspan="4"/>\n                                <footer>\n                                    <button name="config" string="Start Configuration" type="object" class="btn-primary"/>\n                                    <button special="cancel" string="Close" class="btn-default"/>\n                                </footer>\n                             </form>'
        return res

    @api.multi
    def upgrade_module_cancel(self):
        if False:
            print('Hello World!')
        Module = self.env['ir.module.module']
        to_install = Module.search([('state', 'in', ['to upgrade', 'to remove'])])
        to_install.write({'state': 'installed'})
        to_uninstall = Module.search([('state', '=', 'to install')])
        to_uninstall.write({'state': 'uninstalled'})
        return {'type': 'ir.actions.act_window_close'}

    @api.multi
    def upgrade_module(self):
        if False:
            i = 10
            return i + 15
        Module = self.env['ir.module.module']
        mods = Module.search([('state', 'in', ['to upgrade', 'to install'])])
        if mods:
            query = ' SELECT d.name\n                        FROM ir_module_module m\n                        JOIN ir_module_module_dependency d ON (m.id = d.module_id)\n                        LEFT JOIN ir_module_module m2 ON (d.name = m2.name)\n                        WHERE m.id in %s and (m2.state IS NULL or m2.state IN %s) '
            self._cr.execute(query, (tuple(mods.ids), ('uninstalled',)))
            unmet_packages = [row[0] for row in self._cr.fetchall()]
            if unmet_packages:
                raise UserError(_('The following modules are not installed or unknown: %s') % ('\n\n' + '\n'.join(unmet_packages)))
            mods.download()
        self._cr.commit()
        api.Environment.reset()
        odoo.modules.registry.Registry.new(self._cr.dbname, update_module=True)
        return {'type': 'ir.actions.act_window_close'}

    @api.multi
    def config(self):
        if False:
            i = 10
            return i + 15
        return self.env['res.config'].next()
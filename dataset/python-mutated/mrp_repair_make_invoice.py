from odoo import api, fields, models

class MakeInvoice(models.TransientModel):
    _name = 'mrp.repair.make_invoice'
    _description = 'Make Invoice'
    group = fields.Boolean('Group by partner invoice address')

    @api.multi
    def make_invoices(self):
        if False:
            i = 10
            return i + 15
        if not self._context.get('active_ids'):
            return {'type': 'ir.actions.act_window_close'}
        new_invoice = {}
        for wizard in self:
            repairs = self.env['mrp.repair'].browse(self._context['active_ids'])
            new_invoice = repairs.action_invoice_create(group=wizard.group)
            repairs.action_repair_invoice_create()
        return {'domain': [('id', 'in', new_invoice.values())], 'name': 'Invoices', 'view_type': 'form', 'view_mode': 'tree,form', 'res_model': 'account.invoice', 'view_id': False, 'views': [(self.env.ref('account.invoice_tree').id, 'tree'), (self.env.ref('account.invoice_form').id, 'form')], 'context': "{'type':'out_invoice'}", 'type': 'ir.actions.act_window'}
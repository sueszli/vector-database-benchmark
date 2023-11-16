from odoo import api, fields, models

class ProductTemplate(models.Model):
    _inherit = 'product.template'
    can_be_expensed = fields.Boolean(help='Specify whether the product can be selected in an HR expense.', string='Can be Expensed')

    @api.model
    def create(self, vals):
        if False:
            while True:
                i = 10
        if vals.get('can_be_expensed', False):
            vals.update({'supplier_taxes_id': False})
        return super(ProductTemplate, self).create(vals)
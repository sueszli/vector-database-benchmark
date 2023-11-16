from odoo import api, fields, models

@api.model
def referenceable_models(self):
    if False:
        for i in range(10):
            print('nop')
    return [(link.object, link.name) for link in self.env['res.request.link'].search([])]

class ResRequestLink(models.Model):
    _name = 'res.request.link'
    _order = 'priority'
    name = fields.Char(required=True, translate=True)
    object = fields.Char(required=True)
    priority = fields.Integer(default=5)
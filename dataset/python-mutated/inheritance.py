from odoo import models, fields

class Inheritance0(models.Model):
    _name = 'inheritance.0'
    name = fields.Char()

    def call(self):
        if False:
            while True:
                i = 10
        return self.check('model 0')

    def check(self, s):
        if False:
            return 10
        return 'This is {} record {}'.format(s, self.name)

class Inheritance1(models.Model):
    _name = 'inheritance.1'
    _inherit = 'inheritance.0'

    def call(self):
        if False:
            i = 10
            return i + 15
        return self.check('model 1')
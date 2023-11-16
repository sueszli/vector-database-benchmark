from odoo import api, fields, models, tools

class DecimalPrecision(models.Model):
    _name = 'decimal.precision'
    name = fields.Char('Usage', index=True, required=True)
    digits = fields.Integer('Digits', required=True, default=2)
    _sql_constraints = [('name_uniq', 'unique (name)', 'Only one value can be defined for each given usage!')]

    @api.model
    @tools.ormcache('application')
    def precision_get(self, application):
        if False:
            while True:
                i = 10
        self.env.cr.execute('select digits from decimal_precision where name=%s', (application,))
        res = self.env.cr.fetchone()
        return res[0] if res else 2

    @api.model_cr
    def clear_cache(self):
        if False:
            for i in range(10):
                print('nop')
        ' Deprecated, use `clear_caches` instead. '
        self.clear_caches()

    @api.model
    def create(self, data):
        if False:
            for i in range(10):
                print('nop')
        res = super(DecimalPrecision, self).create(data)
        self.clear_caches()
        return res

    @api.multi
    def write(self, data):
        if False:
            print('Hello World!')
        res = super(DecimalPrecision, self).write(data)
        self.clear_caches()
        return res

    @api.multi
    def unlink(self):
        if False:
            print('Hello World!')
        res = super(DecimalPrecision, self).unlink()
        self.clear_caches()
        return res

class DecimalPrecisionFloat(models.AbstractModel):
    """ Override qweb.field.float to add a `decimal_precision` domain option
    and use that instead of the column's own value if it is specified
    """
    _inherit = 'ir.qweb.field.float'

    @api.model
    def precision(self, field, options=None):
        if False:
            i = 10
            return i + 15
        dp = options and options.get('decimal_precision')
        if dp:
            return self.env['decimal.precision'].precision_get(dp)
        return super(DecimalPrecisionFloat, self).precision(field, options=options)

class DecimalPrecisionTestModel(models.Model):
    _name = 'decimal.precision.test'
    float = fields.Float()
    float_2 = fields.Float(digits=(16, 2))
    float_4 = fields.Float(digits=(16, 4))
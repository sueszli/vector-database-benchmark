import base64
from ast import literal_eval
from odoo import models, fields, api
from odoo.tools import pickle as cPickle

class pos_cache(models.Model):
    _name = 'pos.cache'
    cache = fields.Binary(attachment=True)
    product_domain = fields.Text(required=True)
    product_fields = fields.Text(required=True)
    config_id = fields.Many2one('pos.config', ondelete='cascade', required=True)
    compute_user_id = fields.Many2one('res.users', 'Cache compute user', required=True)

    @api.model
    def refresh_all_caches(self):
        if False:
            while True:
                i = 10
        self.env['pos.cache'].search([]).refresh_cache()

    @api.one
    def refresh_cache(self):
        if False:
            i = 10
            return i + 15
        products = self.env['product.product'].search(self.get_product_domain())
        prod_ctx = products.with_context(pricelist=self.config_id.pricelist_id.id, display_default_code=False, lang=self.compute_user_id.lang)
        prod_ctx = prod_ctx.sudo(self.compute_user_id.id)
        res = prod_ctx.read(self.get_product_fields())
        datas = {'cache': base64.encodestring(cPickle.dumps(res))}
        self.write(datas)

    @api.model
    def get_product_domain(self):
        if False:
            return 10
        return literal_eval(self.product_domain)

    @api.model
    def get_product_fields(self):
        if False:
            return 10
        return literal_eval(self.product_fields)

    @api.model
    def get_cache(self, domain, fields):
        if False:
            print('Hello World!')
        if not self.cache or domain != self.get_product_domain() or fields != self.get_product_fields():
            self.product_domain = str(domain)
            self.product_fields = str(fields)
            self.refresh_cache()
        cache = base64.decodestring(self.cache)
        return cPickle.loads(cache)

class pos_config(models.Model):
    _inherit = 'pos.config'

    @api.one
    @api.depends('cache_ids')
    def _get_oldest_cache_time(self):
        if False:
            return 10
        pos_cache = self.env['pos.cache']
        oldest_cache = pos_cache.search([('config_id', '=', self.id)], order='write_date', limit=1)
        if oldest_cache:
            self.oldest_cache_time = oldest_cache.write_date
    cache_ids = fields.One2many('pos.cache', 'config_id')
    oldest_cache_time = fields.Datetime(compute='_get_oldest_cache_time', string='Oldest cache time', readonly=True)

    def _get_cache_for_user(self):
        if False:
            while True:
                i = 10
        pos_cache = self.env['pos.cache']
        cache_for_user = pos_cache.search([('id', 'in', self.cache_ids.ids), ('compute_user_id', '=', self.env.uid)])
        if cache_for_user:
            return cache_for_user[0]
        else:
            return None

    @api.multi
    def get_products_from_cache(self, fields, domain):
        if False:
            for i in range(10):
                print('nop')
        cache_for_user = self._get_cache_for_user()
        if cache_for_user:
            return cache_for_user.get_cache(domain, fields)
        else:
            pos_cache = self.env['pos.cache']
            pos_cache.create({'config_id': self.id, 'product_domain': str(domain), 'product_fields': str(fields), 'compute_user_id': self.env.uid})
            new_cache = self._get_cache_for_user()
            return new_cache.get_cache(domain, fields)

    @api.one
    def delete_cache(self):
        if False:
            for i in range(10):
                print('nop')
        self.cache_ids.unlink()
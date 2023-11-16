import psycopg2
from kamserp_config import SHIPMENT_PAYMENT_ON_DELIVERY_NAME, SHIPMENT_PERSONAL_COLLECTION_NAME, SHIPMENT_PREPAYMENT_NAME, SHIPMENT_INPOST_NAME
from odoo import api, fields, models, tools, _
from odoo.exceptions import except_orm
import logging
_logger = logging.getLogger(__name__)

class KamsERPProductsTemplate(models.Model):
    _name = 'product.template'
    _inherit = 'product.template'
    unique_product_number = fields.Text('Unique Product Number')
    price_subiekt = fields.Float('Shop Price', related='product_variant_ids.price_subiekt')
    _sql_constraints = [('unique_number_uniq', 'unique(unique_product_number)', _('A unique_product_number can only be assigned to one product !'))]

    @api.multi
    def unlink(self):
        if False:
            i = 10
            return i + 15
        stock_quant_obj = self.env['stock.quant']
        for product in self:
            other_product_ids = self.env['product.product'].search([('product_tmpl_id', '=', product.id)])
            for product_product in other_product_ids:
                stock_quant_obj.with_context(force_unlink=True).search([('product_id', '=', product_product.id)]).unlink()
        res = super(KamsERPProductsTemplate, self).unlink()
        return res

    def __remove_from_category(self, cr, uid, categ_id, product_id, context):
        if False:
            i = 10
            return i + 15
        category_obj = self.pool['product.category']
        res = category_obj.write(cr, uid, categ_id, {'product_ids': [(2, product_id)]}, context=context)
        if res:
            try:
                categ_instance = category_obj.read(cr, uid, [categ_id])[0]
                if isinstance(categ_instance, dict):
                    if categ_instance.get('parent_id')[0] > 1 and (not isinstance(categ_instance.get('parent_id')[0], bool)):
                        self.__remove_from_category(cr, uid, categ_instance.get('parent_id')[0], product_id, context)
            except TypeError:
                pass

class KamsERPProducts(models.Model):
    _name = 'product.product'
    _inherit = 'product.product'

    def onchange_type(self):
        if False:
            i = 10
            return i + 15
        return {'value': {}}

    def __get_amount_of_product(self, record):
        if False:
            i = 10
            return i + 15
        total_amount = 0
        r = record.read(['product_tmpl_id'])
        template = self.env['product.template'].browse(r[0]['product_tmpl_id'][0])
        for product_id in template.product_variant_ids:
            total_amount += product_id.qty_available
        return total_amount

    @api.depends('product_tmpl_id')
    def _count_amount_of_attribute_line(self):
        if False:
            i = 10
            return i + 15
        " Count kind's amount of attribute line product the same kind. "
        for record in self:
            record.total_amount_of_attribute_line = self.__get_amount_of_product(record)
    price_kqs = fields.Float(related='list_price', string='Web Price', store=True)
    price_subiekt = fields.Float('Shop Price')
    total_amount_of_attribute_line = fields.Integer(compute='_count_amount_of_attribute_line', method=True, string='Total amount of this kind product')

    def __update_category(self, cr, uid, categ_id, product_id, context):
        if False:
            return 10
        category_obj = self.pool['product.category']
        res = category_obj.write(cr, uid, categ_id, {'product_ids': [(4, product_id)]}, context=context)
        if res:
            try:
                categ_instance = category_obj.read(cr, uid, [categ_id])[0]
                if isinstance(categ_instance, dict):
                    if categ_instance.get('parent_id')[0] > 1 and (not isinstance(categ_instance.get('parent_id')[0], bool)):
                        self.__update_category(cr, uid, categ_instance.get('parent_id')[0], product_id, context)
            except TypeError:
                pass

    @api.multi
    def unlink(self):
        if False:
            while True:
                i = 10
        unlink_products = self.env['product.product']
        unlink_templates = self.env['product.template']
        stock_quant_obj = self.env['stock.quant']
        for product in self:
            if not product.exists():
                continue
            other_products = self.search([('product_tmpl_id', '=', product.product_tmpl_id.id), ('id', '!=', product.id)])
            stock_quant_obj.with_context(force_unlink=True).search([('product_id', '=', product.id)]).unlink()
            if not other_products:
                unlink_templates |= product.product_tmpl_id
            unlink_products |= product
        res = super(KamsERPProducts, unlink_products).unlink()
        unlink_templates.unlink()
        return res

class KamsERPManufacturer(models.Model):
    _name = 'res.partner'
    _inherit = 'res.partner'
    kqs_original_id = fields.Integer('Original KQS id', select=True, help='Gives the original KQS id of manufacturer.')
    product_ids = fields.One2many('product.product', 'product_tmpl_id', ondelete='cascade')

class KamsERPSupplier(models.Model):
    _name = 'product.supplierinfo'
    _inherit = 'product.supplierinfo'
    kqs_original_id = fields.Integer('Original KQS id', select=True, help='Gives the original KQS id of supplier.')

class KamsERPAttribute(models.Model):
    _name = 'product.attribute'
    _inherit = 'product.attribute'
    kqs_original_id = fields.Integer('Original KQS id', select=True, help='Gives the original KQS id of supplier.')

class KamsERPAttributeValue(models.Model):
    _name = 'product.attribute.value'
    _inherit = 'product.attribute.value'
    kqs_original_id = fields.Integer('Original KQS id', select=True, help='Gives the original KQS id of supplier.')

class KamsERPCategory(models.Model):
    _name = 'product.category'
    _inherit = 'product.category'
    kqs_original_id = fields.Integer('Original KQS id', select=True, help='Gives the original KQS id of category.')
    product_ids = (fields.Many2many('product.product', 'product_tmpl_id', string='Products', auto_join=True, ondelete='cascade'),)

class KamsERPOrder(models.Model):
    _name = 'sale.order'
    _inherit = 'sale.order'
    kqs_original_id = (fields.Integer('Original KQS id', select=True, help='Gives the original KQS id of category.'),)
    document_type = fields.Selection([('invoice', 'Invoice'), ('receipt', 'Receipt')], 'Type', required=False)
    unique_number = fields.Char('Unique number')
    customer_ip = fields.Char('customer IP address')
    _sql_constraints = [('unique_number_uniq', 'unique(unique_number)', _('A unique_number can only be assigned to one order !'))]

class KamsERPOrderStatus(models.Model):
    _name = 'sale.order.status'

    @api.multi
    def name_get(self):
        if False:
            for i in range(10):
                print('nop')
        old_status = super(KamsERPOrderStatus, self).name_get()
        data = []
        for name in self:
            if name.name == 'ordered':
                display_value = 'Zamówione'
            elif name.name == 'ready':
                display_value = 'Gotowy'
            elif name.name == 'excepted':
                display_value = 'Spodziewany'
            elif name.name == 'none':
                display_value = 'Brak'
            elif name.name == 'aborted':
                display_value = 'Odrzucono'
            elif name.name == 'complete':
                display_value = 'Zrealizowane'
            else:
                display_value = ''
            data.append((name.id, display_value))
        return data
    name = fields.Selection([('ordered', 'Ordered'), ('ready', 'Ready'), ('excepted', 'Excepted'), ('none', 'None'), ('aborted', 'Aborted'), ('complete', 'Complete')], 'Order Status', required=False)

class KamsERPOrderStatusDate(models.Model):
    _name = 'sale.order.status.date'
    order_status_date = fields.Date(string='Order Date', index=False, copy=False, default=fields.Datetime.now)
    name = fields.Many2one('sale.order.status', string='Order status')

class KamsERPOrderLine(models.Model):
    _name = 'sale.order.line'
    _inherit = 'sale.order.line'
    _inherits = {'sale.order.status.date': 'sale_order_status_date_id'}

    @api.model
    def _get_default_name(self):
        if False:
            return 10
        new_default_record = self.env['sale.order.status.date'].create({'name': False})
        return new_default_record

    @api.depends('product_id')
    def _check_if_is_delivery(self):
        if False:
            i = 10
            return i + 15
        if self.product_id.name == SHIPMENT_PAYMENT_ON_DELIVERY_NAME or self.product_id.name == SHIPMENT_PERSONAL_COLLECTION_NAME or self.product_id.name == SHIPMENT_PREPAYMENT_NAME or (self.product_id.name == SHIPMENT_INPOST_NAME):
            self.is_delivery = True
        else:
            self.is_delivery = False
    is_delivery = fields.Boolean(string='Delivery Options', compute='_check_if_is_delivery', store=True)
    status_id = fields.Many2one('sale.order.status', string='Order status')
    sale_order_status_date_id = fields.Many2one('sale.order.status.date', 'Sale Order Status Date', default=lambda self: self._get_default_name())
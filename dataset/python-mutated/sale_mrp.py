from odoo import api, fields, models
from odoo.tools import float_compare

class MrpProduction(models.Model):
    _inherit = 'mrp.production'
    sale_name = fields.Char(compute='_compute_sale_name_sale_ref', string='Sale Name', help='Indicate the name of sales order.')
    sale_ref = fields.Char(compute='_compute_sale_name_sale_ref', string='Sale Reference', help='Indicate the Customer Reference from sales order.')

    @api.multi
    def _compute_sale_name_sale_ref(self):
        if False:
            while True:
                i = 10

        def get_parent_move(move):
            if False:
                i = 10
                return i + 15
            if move.move_dest_id:
                return get_parent_move(move.move_dest_id)
            return move
        for production in self:
            move = get_parent_move(production.move_finished_ids[0])
            production.sale_name = move.procurement_id and move.procurement_id.sale_line_id and move.procurement_id.sale_line_id.order_id.name or False
            production.sale_ref = move.procurement_id and move.procurement_id.sale_line_id and move.procurement_id.sale_line_id.order_id.client_order_ref or False

class SaleOrderLine(models.Model):
    _inherit = 'sale.order.line'

    @api.multi
    def _get_delivered_qty(self):
        if False:
            while True:
                i = 10
        self.ensure_one()
        precision = self.env['decimal.precision'].precision_get('Product Unit of Measure')
        bom_delivered = {}
        bom = self.env['mrp.bom']._bom_find(product=self.product_id)
        if bom and bom.type == 'phantom':
            bom_delivered[bom.id] = False
            product_uom_qty_bom = self.product_uom._compute_quantity(self.product_uom_qty, bom.product_uom_id) / bom.product_qty
            (boms, lines) = bom.explode(self.product_id, product_uom_qty_bom)
            for (bom_line, data) in lines:
                qty = 0.0
                for move in self.procurement_ids.mapped('move_ids'):
                    if move.state == 'done' and move.product_id.id == bom_line.product_id.id:
                        qty += move.product_uom._compute_quantity(move.product_uom_qty, bom_line.product_uom_id)
                if float_compare(qty, data['qty'], precision_digits=precision) < 0:
                    bom_delivered[bom.id] = False
                    break
                else:
                    bom_delivered[bom.id] = True
        if bom_delivered and any(bom_delivered.values()):
            return self.product_uom_qty
        elif bom_delivered:
            return 0.0
        return super(SaleOrderLine, self)._get_delivered_qty()

    @api.multi
    def _get_bom_component_qty(self, bom):
        if False:
            for i in range(10):
                print('nop')
        bom_quantity = self.product_uom._compute_quantity(self.product_uom_qty, bom.product_uom_id)
        (boms, lines) = bom.explode(self.product_id, bom_quantity)
        components = {}
        for (line, line_data) in lines:
            product = line.product_id.id
            uom = line.product_uom_id
            qty = line.product_qty
            if components.get(product, False):
                if uom.id != components[product]['uom']:
                    from_uom = uom
                    to_uom = self.env['product.uom'].browse(components[product]['uom'])
                    qty = from_uom._compute_quantity(qty, to_uom_id=to_uom)
                components[product]['qty'] += qty
            else:
                to_uom = self.env['product.product'].browse(product).uom_id
                if uom.id != to_uom.id:
                    from_uom = uom
                    qty = from_uom._compute_quantity(qty, to_uom_id=to_uom)
                components[product] = {'qty': qty, 'uom': to_uom.id}
        return components

class AccountInvoiceLine(models.Model):
    _inherit = 'account.invoice.line'

    def _get_anglo_saxon_price_unit(self):
        if False:
            while True:
                i = 10
        price_unit = super(AccountInvoiceLine, self)._get_anglo_saxon_price_unit()
        if self.product_id.invoice_policy == 'delivery':
            for s_line in self.sale_line_ids:
                qty_done = sum([x.uom_id._compute_quantity(x.quantity, x.product_id.uom_id) for x in s_line.invoice_lines if x.invoice_id.state in ('open', 'paid')])
                quantity = self.uom_id._compute_quantity(self.quantity, self.product_id.uom_id)
                moves = self.env['stock.move']
                for procurement in s_line.procurement_ids:
                    moves |= procurement.move_ids
                moves.sorted(lambda x: x.date)
                bom = s_line.product_id.product_tmpl_id.bom_ids and s_line.product_id.product_tmpl_id.bom_ids[0]
                if bom.type == 'phantom':
                    average_price_unit = 0
                    components = s_line._get_bom_component_qty(bom)
                    for product_id in components.keys():
                        factor = components[product_id]['qty']
                        prod_moves = [m for m in moves if m.product_id.id == product_id]
                        prod_qty_done = factor * qty_done
                        prod_quantity = factor * quantity
                        average_price_unit += self._compute_average_price(prod_qty_done, prod_quantity, prod_moves)
                    price_unit = average_price_unit or price_unit
                    price_unit = self.product_id.uom_id._compute_price(price_unit, self.uom_id)
        return price_unit
from odoo.tests.common import TransactionCase

class TestUom(TransactionCase):

    def setUp(self):
        if False:
            return 10
        super(TestUom, self).setUp()
        self.uom_gram = self.env.ref('product.product_uom_gram')
        self.uom_kgm = self.env.ref('product.product_uom_kgm')
        self.uom_ton = self.env.ref('product.product_uom_ton')
        self.uom_unit = self.env.ref('product.product_uom_unit')
        self.uom_dozen = self.env.ref('product.product_uom_dozen')
        self.categ_unit_id = self.ref('product.product_uom_categ_unit')

    def test_10_conversion(self):
        if False:
            print('Hello World!')
        qty = self.uom_gram._compute_quantity(1020000, self.uom_ton)
        self.assertEquals(qty, 1.02, 'Converted quantity does not correspond.')
        price = self.uom_gram._compute_price(2, self.uom_ton)
        self.assertEquals(price, 2000000.0, 'Converted price does not correspond.')
        qty = self.uom_dozen._compute_quantity(1, self.uom_unit)
        self.assertEquals(qty, 12.0, 'Converted quantity does not correspond.')
        self.uom_gram.write({'rounding': 1})
        qty = self.uom_gram._compute_quantity(1234, self.uom_kgm)
        self.assertEquals(qty, 1.234, 'Converted quantity does not correspond.')

    def test_20_rounding(self):
        if False:
            while True:
                i = 10
        product_uom = self.env['product.uom'].create({'name': 'Score', 'factor_inv': 20, 'uom_type': 'bigger', 'rounding': 1.0, 'category_id': self.categ_unit_id})
        qty = self.uom_unit._compute_quantity(2, product_uom)
        self.assertEquals(qty, 1, 'Converted quantity should be rounded up.')
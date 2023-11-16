from odoo.addons.procurement.tests.common import TestStockCommon

class TestBase(TestStockCommon):

    def test_base(self):
        if False:
            for i in range(10):
                print('nop')
        procurement = self._create_procurement(self.user_employee, product_id=self.product_1.id, name='Procurement Test', product_qty=15.0)
        self.assertEqual(procurement.state, 'exception')
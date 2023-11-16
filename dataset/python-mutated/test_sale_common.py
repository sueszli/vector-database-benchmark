from odoo.addons.account.tests.account_test_classes import AccountingTestCase

class TestSale(AccountingTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestSale, self).setUp()
        group_manager = self.env.ref('sales_team.group_sale_manager')
        group_user = self.env.ref('sales_team.group_sale_salesman')
        self.manager = self.env['res.users'].create({'name': 'Andrew Manager', 'login': 'manager', 'email': 'a.m@example.com', 'signature': '--\nAndreww', 'notify_email': 'always', 'groups_id': [(6, 0, [group_manager.id])]})
        self.user = self.env['res.users'].create({'name': 'Mark User', 'login': 'user', 'email': 'm.u@example.com', 'signature': '--\nMark', 'notify_email': 'always', 'groups_id': [(6, 0, [group_user.id])]})
        self.products = {'prod_order': self.env.ref('product.product_order_01'), 'prod_del': self.env.ref('product.product_delivery_01'), 'serv_order': self.env.ref('product.service_order_01'), 'serv_del': self.env.ref('product.service_delivery')}
        self.partner = self.env.ref('base.res_partner_1')
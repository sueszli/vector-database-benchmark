import odoo.tests

@odoo.tests.common.at_install(False)
@odoo.tests.common.post_install(True)
class TestUi(odoo.tests.HttpCase):

    def test_01_admin_shop_tour(self):
        if False:
            return 10
        self.phantom_js('/', "odoo.__DEBUG__.services['web_tour.tour'].run('shop')", "odoo.__DEBUG__.services['web_tour.tour'].tours.shop.ready", login='admin')

    def test_02_admin_checkout(self):
        if False:
            print('Hello World!')
        self.phantom_js('/', "odoo.__DEBUG__.services['web_tour.tour'].run('shop_buy_product')", "odoo.__DEBUG__.services['web_tour.tour'].tours.shop_buy_product.ready", login='admin')

    def test_03_demo_checkout(self):
        if False:
            return 10
        self.phantom_js('/', "odoo.__DEBUG__.services['web_tour.tour'].run('shop_buy_product')", "odoo.__DEBUG__.services['web_tour.tour'].tours.shop_buy_product.ready", login='demo')
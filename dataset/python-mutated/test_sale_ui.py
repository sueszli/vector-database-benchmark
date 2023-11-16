import odoo.tests

@odoo.tests.common.at_install(False)
@odoo.tests.common.post_install(True)
class TestUi(odoo.tests.HttpCase):

    def test_01_sale_tour(self):
        if False:
            for i in range(10):
                print('nop')
        self.phantom_js('/web', "odoo.__DEBUG__.services['web_tour.tour'].run('sale_tour')", "odoo.__DEBUG__.services['web_tour.tour'].tours.sale_tour.ready", login='admin')
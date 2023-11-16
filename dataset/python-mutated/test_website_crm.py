from odoo.api import Environment
import odoo.tests

@odoo.tests.common.at_install(False)
@odoo.tests.common.post_install(True)
class TestWebsiteCrm(odoo.tests.HttpCase):

    def test_tour(self):
        if False:
            for i in range(10):
                print('nop')
        self.phantom_js('/', "odoo.__DEBUG__.services['web_tour.tour'].run('website_crm_tour')", "odoo.__DEBUG__.services['web_tour.tour'].tours.website_crm_tour.ready")
        cr = self.registry.cursor()
        assert cr is self.registry.test_cr
        env = Environment(cr, self.uid, {})
        record = env['crm.lead'].search([('description', '=', '### TOUR DATA ###')])
        assert len(record) == 1
        assert record.contact_name == 'John Smith'
        assert record.email_from == 'john@smith.com'
        assert record.partner_name == 'Odoo S.A.'
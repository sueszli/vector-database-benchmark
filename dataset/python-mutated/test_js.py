import unittest
import odoo.tests

class WebSuite(odoo.tests.HttpCase):

    @unittest.skip('Memory leak in this test lead to phantomjs crash, making it unreliable')
    def test_01_js(self):
        if False:
            for i in range(10):
                print('nop')
        self.phantom_js('/web/tests?mod=web', '', '', login='admin')
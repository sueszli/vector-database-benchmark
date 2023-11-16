from odoo.tests import common

class TestBasicInheritance(common.TransactionCase):

    def test_inherit_method(self):
        if False:
            while True:
                i = 10
        env = self.env
        a = env['inheritance.0'].create({'name': 'A'})
        b = env['inheritance.1'].create({'name': 'B'})
        self.assertEqual(a.call(), 'This is model 0 record A')
        self.assertEqual(b.call(), 'This is model 1 record B')
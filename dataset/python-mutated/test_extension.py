from odoo.tests import common

class TestBasicInheritance(common.TransactionCase):

    def test_extend_fields(self):
        if False:
            while True:
                i = 10
        env = self.env
        record = env['extension.0'].create({})
        self.assertDictContainsSubset({'name': 'A', 'description': 'Extended'}, record.read()[0])
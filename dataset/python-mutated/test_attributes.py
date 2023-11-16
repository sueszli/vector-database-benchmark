from odoo.tests import common
ANSWER_TO_ULTIMATE_QUESTION = 42

class TestAttributes(common.TransactionCase):

    def test_we_can_add_attributes(self):
        if False:
            while True:
                i = 10
        Model = self.env['test_new_api.category']
        instance = Model.create({'name': 'Foo'})
        instance.unknown = ANSWER_TO_ULTIMATE_QUESTION
        self.assertTrue(hasattr(instance, 'unknown'))
        self.assertIsInstance(instance.unknown, (int, long))
        self.assertEqual(instance.unknown, ANSWER_TO_ULTIMATE_QUESTION)
        self.assertEqual(getattr(instance, 'unknown'), ANSWER_TO_ULTIMATE_QUESTION)
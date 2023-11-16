import st2common.bootstrap.ruletypesregistrar as ruletypes_registrar
from st2tests.api import FunctionalTest

class TestRuleTypesController(FunctionalTest):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(TestRuleTypesController, cls).setUpClass()
        ruletypes_registrar.register_rule_types()

    def test_get_one(self):
        if False:
            for i in range(10):
                print('nop')
        list_resp = self.app.get('/v1/ruletypes')
        self.assertEqual(list_resp.status_int, 200)
        self.assertTrue(len(list_resp.json) > 0, '/v1/ruletypes did not return correct ruletypes.')
        ruletype_id = list_resp.json[0]['id']
        get_resp = self.app.get('/v1/ruletypes/%s' % ruletype_id)
        retrieved_id = get_resp.json['id']
        self.assertEqual(get_resp.status_int, 200)
        self.assertEqual(retrieved_id, ruletype_id, '/v1/ruletypes returned incorrect ruletype.')

    def test_get_all(self):
        if False:
            return 10
        resp = self.app.get('/v1/ruletypes')
        self.assertEqual(resp.status_int, 200)
        self.assertTrue(len(resp.json) > 0, '/v1/ruletypes did not return correct ruletypes.')

    def test_get_one_fail_doesnt_exist(self):
        if False:
            print('Hello World!')
        resp = self.app.get('/v1/ruletypes/1', expect_errors=True)
        self.assertEqual(resp.status_int, 404)
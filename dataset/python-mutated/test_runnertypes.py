from st2common.util.monkey_patch import monkey_patch
monkey_patch()
from st2api.controllers.v1.runnertypes import RunnerTypesController
from st2tests.api import FunctionalTest
from st2tests.api import APIControllerWithIncludeAndExcludeFilterTestCase
__all__ = ['RunnerTypesControllerTestCase']

class RunnerTypesControllerTestCase(FunctionalTest, APIControllerWithIncludeAndExcludeFilterTestCase):
    get_all_path = '/v1/runnertypes'
    controller_cls = RunnerTypesController
    include_attribute_field_name = 'runner_package'
    exclude_attribute_field_name = 'runner_module'
    test_exact_object_count = False

    def test_get_one(self):
        if False:
            while True:
                i = 10
        resp = self.app.get('/v1/runnertypes')
        self.assertEqual(resp.status_int, 200)
        self.assertTrue(len(resp.json) > 0, '/v1/runnertypes did not return correct runnertypes.')
        runnertype_id = RunnerTypesControllerTestCase.__get_runnertype_id(resp.json[0])
        resp = self.app.get('/v1/runnertypes/%s' % runnertype_id)
        retrieved_id = RunnerTypesControllerTestCase.__get_runnertype_id(resp.json)
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(retrieved_id, runnertype_id, '/v1/runnertypes returned incorrect runnertype.')

    def test_get_all(self):
        if False:
            print('Hello World!')
        resp = self.app.get('/v1/runnertypes')
        self.assertEqual(resp.status_int, 200)
        self.assertTrue(len(resp.json) > 0, '/v1/runnertypes did not return correct runnertypes.')

    def test_get_one_fail_doesnt_exist(self):
        if False:
            i = 10
            return i + 15
        resp = self.app.get('/v1/runnertypes/1', expect_errors=True)
        self.assertEqual(resp.status_int, 404)

    def test_put_disable_runner(self):
        if False:
            print('Hello World!')
        runnertype_id = 'action-chain'
        resp = self.app.get('/v1/runnertypes/%s' % runnertype_id)
        self.assertTrue(resp.json['enabled'])
        update_input = resp.json
        update_input['enabled'] = False
        update_input['name'] = 'foobar'
        put_resp = self.__do_put(runnertype_id, update_input)
        self.assertFalse(put_resp.json['enabled'])
        self.assertEqual(put_resp.json['name'], 'action-chain')
        update_input = resp.json
        update_input['enabled'] = True
        put_resp = self.__do_put(runnertype_id, update_input)
        self.assertTrue(put_resp.json['enabled'])

    def __do_put(self, runner_type_id, runner_type):
        if False:
            print('Hello World!')
        return self.app.put_json('/v1/runnertypes/%s' % runner_type_id, runner_type, expect_errors=True)

    @staticmethod
    def __get_runnertype_id(resp_json):
        if False:
            i = 10
            return i + 15
        return resp_json['id']
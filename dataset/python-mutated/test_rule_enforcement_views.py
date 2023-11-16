import six
from st2api.controllers.v1.rule_enforcement_views import RuleEnforcementViewController
from st2tests.fixtures.rule_enforcements.fixture import PACK_NAME as FIXTURES_PACK
from st2tests.fixturesloader import FixturesLoader
from st2tests.api import FunctionalTest
from st2tests.api import APIControllerWithIncludeAndExcludeFilterTestCase
__all__ = ['RuleEnforcementViewsControllerTestCase']
http_client = six.moves.http_client
TEST_FIXTURES = {'enforcements': ['enforcement1.yaml', 'enforcement2.yaml', 'enforcement3.yaml'], 'executions': ['execution1.yaml'], 'triggerinstances': ['trigger_instance_1.yaml']}

class RuleEnforcementViewsControllerTestCase(FunctionalTest, APIControllerWithIncludeAndExcludeFilterTestCase):
    get_all_path = '/v1/ruleenforcements/views'
    controller_cls = RuleEnforcementViewController
    include_attribute_field_name = 'enforced_at'
    exclude_attribute_field_name = 'status'
    fixtures_loader = FixturesLoader()

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super(RuleEnforcementViewsControllerTestCase, cls).setUpClass()
        cls.models = RuleEnforcementViewsControllerTestCase.fixtures_loader.save_fixtures_to_db(fixtures_pack=FIXTURES_PACK, fixtures_dict=TEST_FIXTURES, use_object_ids=True)
        cls.ENFORCEMENT_1 = cls.models['enforcements']['enforcement1.yaml']

    def test_get_all(self):
        if False:
            i = 10
            return i + 15
        resp = self.app.get('/v1/ruleenforcements/views')
        self.assertEqual(resp.status_int, http_client.OK)
        self.assertEqual(len(resp.json), 3)
        self.assertEqual(resp.json[0]['trigger_instance']['id'], '565e15ce32ed350857dfa623')
        self.assertEqual(resp.json[0]['trigger_instance']['payload'], {'foo': 'bar', 'name': 'Joe'})
        self.assertEqual(resp.json[0]['execution']['action']['ref'], 'core.local')
        self.assertEqual(resp.json[0]['execution']['action']['parameters'], {'sudo': {'immutable': True}})
        self.assertEqual(resp.json[0]['execution']['runner']['name'], 'action-chain')
        self.assertEqual(resp.json[0]['execution']['runner']['runner_parameters'], {'foo': {'type': 'string'}})
        self.assertEqual(resp.json[0]['execution']['parameters'], {'cmd': 'echo bar'})
        self.assertEqual(resp.json[0]['execution']['status'], 'scheduled')
        self.assertEqual(resp.json[1]['trigger_instance'], {})
        self.assertEqual(resp.json[1]['execution'], {})
        self.assertEqual(resp.json[2]['trigger_instance'], {})
        self.assertEqual(resp.json[2]['execution'], {})

    def test_filter_by_rule_ref(self):
        if False:
            while True:
                i = 10
        resp = self.app.get('/v1/ruleenforcements/views?rule_ref=wolfpack.golden_rule')
        self.assertEqual(resp.status_int, http_client.OK)
        self.assertEqual(len(resp.json), 1)
        self.assertEqual(resp.json[0]['rule']['ref'], 'wolfpack.golden_rule')

    def test_get_one_success(self):
        if False:
            i = 10
            return i + 15
        resp = self.app.get('/v1/ruleenforcements/views/%s' % str(self.ENFORCEMENT_1.id))
        self.assertEqual(resp.json['id'], str(self.ENFORCEMENT_1.id))
        self.assertEqual(resp.json['trigger_instance']['id'], '565e15ce32ed350857dfa623')
        self.assertEqual(resp.json['trigger_instance']['payload'], {'foo': 'bar', 'name': 'Joe'})
        self.assertEqual(resp.json['execution']['action']['ref'], 'core.local')
        self.assertEqual(resp.json['execution']['action']['parameters'], {'sudo': {'immutable': True}})
        self.assertEqual(resp.json['execution']['runner']['name'], 'action-chain')
        self.assertEqual(resp.json['execution']['runner']['runner_parameters'], {'foo': {'type': 'string'}})
        self.assertEqual(resp.json['execution']['parameters'], {'cmd': 'echo bar'})
        self.assertEqual(resp.json['execution']['status'], 'scheduled')

    def _insert_mock_models(self):
        if False:
            for i in range(10):
                print('nop')
        enfrocement_ids = [enforcement['id'] for enforcement in self.models['enforcements'].values()]
        return enfrocement_ids

    def _delete_mock_models(self, object_ids):
        if False:
            return 10
        pass
import six
from st2tests.fixtures.descendants.fixture import PACK_NAME as DESCENDANTS_PACK
from st2tests.fixturesloader import FixturesLoader
from st2tests.api import FunctionalTest
DESCENDANTS_FIXTURES = {'executions': ['root_execution.yaml', 'child1_level1.yaml', 'child2_level1.yaml', 'child1_level2.yaml', 'child2_level2.yaml', 'child3_level2.yaml', 'child1_level3.yaml', 'child2_level3.yaml', 'child3_level3.yaml']}

class ActionExecutionControllerTestCaseDescendantsTest(FunctionalTest):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super(ActionExecutionControllerTestCaseDescendantsTest, cls).setUpClass()
        cls.MODELS = FixturesLoader().save_fixtures_to_db(fixtures_pack=DESCENDANTS_PACK, fixtures_dict=DESCENDANTS_FIXTURES)

    def test_get_all_descendants(self):
        if False:
            print('Hello World!')
        root_execution = self.MODELS['executions']['root_execution.yaml']
        resp = self.app.get('/v1/executions/%s/children' % str(root_execution.id))
        self.assertEqual(resp.status_int, 200)
        all_descendants_ids = [descendant['id'] for descendant in resp.json]
        all_descendants_ids.sort()
        expected_ids = [str(v.id) for (_, v) in six.iteritems(self.MODELS['executions']) if v.id != root_execution.id]
        expected_ids.sort()
        self.assertListEqual(all_descendants_ids, expected_ids)

    def test_get_all_descendants_depth_neg_1(self):
        if False:
            return 10
        root_execution = self.MODELS['executions']['root_execution.yaml']
        resp = self.app.get('/v1/executions/%s/children?depth=-1' % str(root_execution.id))
        self.assertEqual(resp.status_int, 200)
        all_descendants_ids = [descendant['id'] for descendant in resp.json]
        all_descendants_ids.sort()
        expected_ids = [str(v.id) for (_, v) in six.iteritems(self.MODELS['executions']) if v.id != root_execution.id]
        expected_ids.sort()
        self.assertListEqual(all_descendants_ids, expected_ids)

    def test_get_1_level_descendants(self):
        if False:
            for i in range(10):
                print('nop')
        root_execution = self.MODELS['executions']['root_execution.yaml']
        resp = self.app.get('/v1/executions/%s/children?depth=1' % str(root_execution.id))
        self.assertEqual(resp.status_int, 200)
        all_descendants_ids = [descendant['id'] for descendant in resp.json]
        all_descendants_ids.sort()
        expected_ids = [str(v.id) for (_, v) in six.iteritems(self.MODELS['executions']) if v.parent == str(root_execution.id)]
        expected_ids.sort()
        self.assertListEqual(all_descendants_ids, expected_ids)
from __future__ import absolute_import
from st2common.exceptions.trace import TraceNotFoundException
from st2common.persistence.liveaction import LiveAction
from st2common.persistence.trace import Trace
import st2common.services.action as action_services
from st2tests.fixtures.traces.fixture import PACK_NAME as FIXTURES_PACK
from st2tests.fixturesloader import FixturesLoader
from st2tests import DbTestCase
TEST_MODELS = {'executions': ['traceable_execution.yaml'], 'liveactions': ['traceable_liveaction.yaml'], 'actions': ['chain1.yaml'], 'runners': ['actionchain.yaml']}

class TraceInjectionTests(DbTestCase):
    models = None
    traceable_liveaction = None
    traceable_execution = None
    action = None

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(TraceInjectionTests, cls).setUpClass()
        cls.models = FixturesLoader().save_fixtures_to_db(fixtures_pack=FIXTURES_PACK, fixtures_dict=TEST_MODELS)
        cls.traceable_liveaction = cls.models['liveactions']['traceable_liveaction.yaml']
        cls.traceable_execution = cls.models['executions']['traceable_execution.yaml']
        cls.action = cls.models['actions']['chain1.yaml']

    def test_trace_provided(self):
        if False:
            print('Hello World!')
        self.traceable_liveaction['context']['trace_context'] = {'trace_tag': 'OohLaLaLa'}
        action_services.request(self.traceable_liveaction)
        traces = Trace.get_all()
        self.assertEqual(len(traces), 1)
        self.assertEqual(len(traces[0]['action_executions']), 1)
        trace_id = str(traces[0].id)
        self.traceable_liveaction['context']['trace_context'] = {'id_': trace_id}
        action_services.request(self.traceable_liveaction)
        traces = Trace.get_all()
        self.assertEqual(len(traces), 1)
        self.assertEqual(len(traces[0]['action_executions']), 2)

    def test_trace_tag_resuse(self):
        if False:
            return 10
        self.traceable_liveaction['context']['trace_context'] = {'trace_tag': 'blank space'}
        action_services.request(self.traceable_liveaction)
        action_services.request(self.traceable_liveaction)
        traces = Trace.query(**{'trace_tag': 'blank space'})
        self.assertEqual(len(traces), 2)

    def test_invalid_trace_id_provided(self):
        if False:
            return 10
        liveactions = LiveAction.get_all()
        self.assertEqual(len(liveactions), 1)
        self.traceable_liveaction['context']['trace_context'] = {'id_': 'balleilaka'}
        self.assertRaises(TraceNotFoundException, action_services.request, self.traceable_liveaction)
        liveactions = LiveAction.get_all()
        self.assertEqual(len(liveactions), 0)
from __future__ import absolute_import
from datetime import timedelta
import bson
from st2common import log as logging
from st2common.garbage_collection.trace import purge_traces
from st2common.models.db.trace import TraceDB, TraceComponentDB
from st2common.persistence.trace import Trace
from st2common.util import date as date_utils
from st2tests.base import CleanDbTestCase
LOG = logging.getLogger(__name__)

class TestPurgeTrace(CleanDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        CleanDbTestCase.setUpClass()
        super(TestPurgeTrace, cls).setUpClass()

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestPurgeTrace, self).setUp()

    def test_no_timestamp_doesnt_delete(self):
        if False:
            while True:
                i = 10
        now = date_utils.get_datetime_utc_now()
        TestPurgeTrace._create_save_trace(trace_tag='test_trace', action_executions=[str(bson.ObjectId()) for _ in range(4)], rules=[str(bson.ObjectId()) for _ in range(4)], trigger_instances=[str(bson.ObjectId()) for _ in range(5)], start_timestamp=now - timedelta(days=20))
        self.assertEqual(len(Trace.get_all()), 1)
        expected_msg = 'Specify a valid timestamp'
        self.assertRaisesRegexp(ValueError, expected_msg, purge_traces, logger=LOG, timestamp=None)
        self.assertEqual(len(Trace.get_all()), 1)

    def test_purge(self):
        if False:
            return 10
        now = date_utils.get_datetime_utc_now()
        TestPurgeTrace._create_save_trace(trace_tag='test_trace', action_executions=[str(bson.ObjectId()) for _ in range(4)], rules=[str(bson.ObjectId()) for _ in range(4)], trigger_instances=[str(bson.ObjectId()) for _ in range(5)], start_timestamp=now - timedelta(days=20))
        TestPurgeTrace._create_save_trace(trace_tag='test_trace', action_executions=[str(bson.ObjectId()) for _ in range(4)], rules=[str(bson.ObjectId()) for _ in range(4)], trigger_instances=[str(bson.ObjectId()) for _ in range(5)], start_timestamp=now - timedelta(days=5))
        self.assertEqual(len(Trace.get_all()), 2)
        purge_traces(logger=LOG, timestamp=now - timedelta(days=10))
        self.assertEqual(len(Trace.get_all()), 1)

    @staticmethod
    def _create_save_trace(trace_tag, id_=None, action_executions=None, rules=None, trigger_instances=None, start_timestamp=None):
        if False:
            i = 10
            return i + 15
        if action_executions is None:
            action_executions = []
        action_executions = [TraceComponentDB(object_id=action_execution) for action_execution in action_executions]
        if rules is None:
            rules = []
        rules = [TraceComponentDB(object_id=rule) for rule in rules]
        if trigger_instances is None:
            trigger_instances = []
        trigger_instances = [TraceComponentDB(object_id=trigger_instance) for trigger_instance in trigger_instances]
        created = TraceDB(id=id_, trace_tag=trace_tag, trigger_instances=trigger_instances, rules=rules, action_executions=action_executions, start_timestamp=start_timestamp)
        return Trace.add_or_update(created)
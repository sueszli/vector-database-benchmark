from __future__ import absolute_import
import mock
from st2common.transport.publishers import PoolPublisher
from st2reactor.container.utils import create_trigger_instance
from st2common.persistence.trigger import Trigger
from st2common.models.db.trigger import TriggerDB
from st2tests.base import CleanDbTestCase

@mock.patch.object(PoolPublisher, 'publish', mock.MagicMock())
class ContainerUtilsTest(CleanDbTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(ContainerUtilsTest, self).setUp()
        trigger_db = TriggerDB(name='name1', pack='pack1', type='type1', parameters={'a': 1, 'b': '2', 'c': 'foo'})
        self.trigger_db = Trigger.add_or_update(trigger_db)

    def test_create_trigger_instance_invalid_trigger(self):
        if False:
            return 10
        trigger_instance = 'dummy_pack.footrigger'
        instance = create_trigger_instance(trigger=trigger_instance, payload={}, occurrence_time=None)
        self.assertIsNone(instance)

    def test_create_trigger_instance_success(self):
        if False:
            i = 10
            return i + 15
        payload = {}
        occurrence_time = None
        trigger = {'id': self.trigger_db.id}
        trigger_instance_db = create_trigger_instance(trigger=trigger, payload=payload, occurrence_time=occurrence_time)
        self.assertEqual(trigger_instance_db.trigger, 'pack1.name1')
        trigger = {'id': '5776aa2b0640fd2991b15987'}
        trigger_instance_db = create_trigger_instance(trigger=trigger, payload=payload, occurrence_time=occurrence_time)
        self.assertEqual(trigger_instance_db, None)
        trigger = {'uid': self.trigger_db.uid}
        trigger_instance_db = create_trigger_instance(trigger=trigger, payload=payload, occurrence_time=occurrence_time)
        self.assertEqual(trigger_instance_db.trigger, 'pack1.name1')
        trigger = {'uid': 'invaliduid'}
        trigger_instance_db = create_trigger_instance(trigger=trigger, payload=payload, occurrence_time=occurrence_time)
        self.assertEqual(trigger_instance_db, None)
        trigger = {'type': 'pack1.name1', 'parameters': self.trigger_db.parameters}
        trigger_instance_db = create_trigger_instance(trigger=trigger, payload=payload, occurrence_time=occurrence_time)
        trigger = {'type': 'pack1.name1', 'parameters': {}}
        trigger_instance_db = create_trigger_instance(trigger=trigger, payload=payload, occurrence_time=occurrence_time)
        self.assertEqual(trigger_instance_db, None)
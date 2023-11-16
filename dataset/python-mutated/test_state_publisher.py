from __future__ import absolute_import
from st2common.util.monkey_patch import monkey_patch
monkey_patch()
import kombu
import mock
import mongoengine as me
from st2common.models import db
from st2common.models.db import stormbase
from st2common.persistence import base as persistence
from st2common.transport import publishers
from st2tests import DbTestCase
FAKE_STATE_MGMT_XCHG = kombu.Exchange('st2.fake.state', type='topic')

class FakeModelPublisher(publishers.StatePublisherMixin):

    def __init__(self):
        if False:
            return 10
        super(FakeModelPublisher, self).__init__(exchange=FAKE_STATE_MGMT_XCHG)

class FakeModelDB(stormbase.StormBaseDB):
    state = me.StringField(required=True)

class FakeModel(persistence.Access):
    impl = db.MongoDBAccess(FakeModelDB)
    publisher = None

    @classmethod
    def _get_impl(cls):
        if False:
            print('Hello World!')
        return cls.impl

    @classmethod
    def _get_publisher(cls):
        if False:
            print('Hello World!')
        if not cls.publisher:
            cls.publisher = FakeModelPublisher()
        return cls.publisher

    @classmethod
    def publish_state(cls, model_object):
        if False:
            print('Hello World!')
        publisher = cls._get_publisher()
        if publisher:
            publisher.publish_state(model_object, getattr(model_object, 'state', None))

    @classmethod
    def _get_by_object(cls, object):
        if False:
            while True:
                i = 10
        return None

class StatePublisherTest(DbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super(StatePublisherTest, cls).setUpClass()
        cls.access = FakeModel()

    def tearDown(self):
        if False:
            while True:
                i = 10
        FakeModelDB.drop_collection()
        super(StatePublisherTest, self).tearDown()

    @mock.patch.object(publishers.PoolPublisher, 'publish', mock.MagicMock())
    def test_publish(self):
        if False:
            for i in range(10):
                print('nop')
        instance = FakeModelDB(state='faked')
        self.access.publish_state(instance)
        publishers.PoolPublisher.publish.assert_called_with(instance, FAKE_STATE_MGMT_XCHG, instance.state)

    def test_publish_unset(self):
        if False:
            return 10
        instance = FakeModelDB()
        self.assertRaises(Exception, self.access.publish_state, instance)

    def test_publish_none(self):
        if False:
            print('Hello World!')
        instance = FakeModelDB(state=None)
        self.assertRaises(Exception, self.access.publish_state, instance)

    def test_publish_empty_str(self):
        if False:
            while True:
                i = 10
        instance = FakeModelDB(state='')
        self.assertRaises(Exception, self.access.publish_state, instance)
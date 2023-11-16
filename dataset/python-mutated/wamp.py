import json
from autobahn.wamp.exception import TransportLost
from autobahn.wamp.types import PublishOptions
from autobahn.wamp.types import SubscribeOptions
from twisted.internet import defer
from twisted.python import log
from buildbot.mq import base
from buildbot.util import service
from buildbot.util import toJson

class WampMQ(service.ReconfigurableServiceMixin, base.MQBase):
    NAMESPACE = 'org.buildbot.mq'

    def produce(self, routingKey, data):
        if False:
            while True:
                i = 10
        d = self._produce(routingKey, data)
        d.addErrback(log.err, 'Problem while producing message on topic ' + repr(routingKey))

    @classmethod
    def messageTopic(cls, routingKey):
        if False:
            print('Hello World!')

        def ifNone(v, default):
            if False:
                for i in range(10):
                    print('nop')
            return default if v is None else v
        routingKey = [ifNone(key, '') for key in routingKey]
        return cls.NAMESPACE + '.' + '.'.join(routingKey)

    @classmethod
    def routingKeyFromMessageTopic(cls, topic):
        if False:
            while True:
                i = 10
        return tuple(topic[len(WampMQ.NAMESPACE) + 1:].split('.'))

    def _produce(self, routingKey, data):
        if False:
            while True:
                i = 10
        _data = json.loads(json.dumps(data, default=toJson))
        options = PublishOptions(exclude_me=False)
        return self.master.wamp.publish(self.messageTopic(routingKey), _data, options=options)

    def startConsuming(self, callback, _filter, persistent_name=None):
        if False:
            i = 10
            return i + 15
        if persistent_name is not None:
            log.err(f'wampmq: persistent queues are not persisted: {persistent_name} {_filter}')
        qr = QueueRef(self, callback)
        self._startConsuming(qr, callback, _filter)
        return defer.succeed(qr)

    def _startConsuming(self, qr, callback, _filter, persistent_name=None):
        if False:
            i = 10
            return i + 15
        return qr.subscribe(self.master.wamp, self, _filter)

class QueueRef(base.QueueRef):

    def __init__(self, mq, callback):
        if False:
            while True:
                i = 10
        super().__init__(callback)
        self.unreg = None
        self.mq = mq

    @defer.inlineCallbacks
    def subscribe(self, connector_service, wamp_service, _filter):
        if False:
            i = 10
            return i + 15
        self.filter = _filter
        self.emulated = False
        options = {'details_arg': str('details')}
        if None in _filter:
            options['match'] = 'wildcard'
        options = SubscribeOptions(**options)
        _filter = WampMQ.messageTopic(_filter)
        self.unreg = (yield connector_service.subscribe(self.wampInvoke, _filter, options=options))
        if self.callback is None:
            yield self.stopConsuming()

    def wampInvoke(self, msg, details):
        if False:
            print('Hello World!')
        if details.topic is not None:
            topic = WampMQ.routingKeyFromMessageTopic(details.topic)
        else:
            topic = self.filter
        self.mq.invokeQref(self, topic, msg)

    @defer.inlineCallbacks
    def stopConsuming(self):
        if False:
            i = 10
            return i + 15
        self.callback = None
        if self.unreg is not None:
            unreg = self.unreg
            self.unreg = None
            try:
                yield unreg.unsubscribe()
            except TransportLost:
                pass
            except Exception as e:
                log.err(e, 'When unsubscribing MQ connection ' + str(unreg))
from twisted.internet import defer
from twisted.python import failure
from zope.interface import implementer
from buildbot.interfaces import ITriggerableScheduler
from buildbot.process.properties import Properties
from buildbot.schedulers import base
from buildbot.util import debounce

@implementer(ITriggerableScheduler)
class Triggerable(base.BaseScheduler):
    compare_attrs = base.BaseScheduler.compare_attrs + ('reason',)

    def __init__(self, name, builderNames, reason=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(name, builderNames, **kwargs)
        self._waiters = {}
        self._buildset_complete_consumer = None
        self.reason = reason

    def trigger(self, waited_for, sourcestamps=None, set_props=None, parent_buildid=None, parent_relationship=None):
        if False:
            return 10
        'Trigger this scheduler with the optional given list of sourcestamps\n        Returns two deferreds:\n            idsDeferred -- yields the ids of the buildset and buildrequest, as soon as they are\n            available.\n            resultsDeferred -- yields the build result(s), when they finish.'
        props = Properties()
        props.updateFromProperties(self.properties)
        reason = self.reason
        if set_props:
            props.updateFromProperties(set_props)
            reason = set_props.getProperty('reason')
        if reason is None:
            reason = f"The Triggerable scheduler named '{self.name}' triggered this build"
        idsDeferred = self.addBuildsetForSourceStampsWithDefaults(reason, sourcestamps, waited_for, priority=self.priority, properties=props, parent_buildid=parent_buildid, parent_relationship=parent_relationship)
        resultsDeferred = defer.Deferred()

        @idsDeferred.addCallback
        def setup_waiter(ids):
            if False:
                print('Hello World!')
            (bsid, brids) = ids
            self._waiters[bsid] = (resultsDeferred, brids)
            self._updateWaiters()
            return ids
        return (idsDeferred, resultsDeferred)

    @defer.inlineCallbacks
    def startService(self):
        if False:
            for i in range(10):
                print('nop')
        yield super().startService()
        self._updateWaiters.start()

    @defer.inlineCallbacks
    def stopService(self):
        if False:
            for i in range(10):
                print('nop')
        yield self._updateWaiters.stop()
        if self._buildset_complete_consumer:
            self._buildset_complete_consumer.stopConsuming()
            self._buildset_complete_consumer = None
        if self._waiters:
            msg = 'Triggerable scheduler stopped before build was complete'
            for (d, _) in self._waiters.values():
                d.errback(failure.Failure(RuntimeError(msg)))
            self._waiters = {}
        yield super().stopService()

    @debounce.method(wait=0)
    @defer.inlineCallbacks
    def _updateWaiters(self):
        if False:
            i = 10
            return i + 15
        if self._waiters and (not self._buildset_complete_consumer):
            startConsuming = self.master.mq.startConsuming
            self._buildset_complete_consumer = (yield startConsuming(self._buildset_complete_cb, ('buildsets', None, 'complete')))
        elif not self._waiters and self._buildset_complete_consumer:
            self._buildset_complete_consumer.stopConsuming()
            self._buildset_complete_consumer = None

    def _buildset_complete_cb(self, key, msg):
        if False:
            for i in range(10):
                print('nop')
        if msg['bsid'] not in self._waiters:
            return
        (d, brids) = self._waiters.pop(msg['bsid'])
        self._updateWaiters()
        d.callback((msg['results'], brids))
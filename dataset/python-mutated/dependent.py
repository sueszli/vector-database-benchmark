from twisted.internet import defer
from buildbot import config
from buildbot import interfaces
from buildbot import util
from buildbot.process.results import SUCCESS
from buildbot.process.results import WARNINGS
from buildbot.schedulers import base

class Dependent(base.BaseScheduler):
    compare_attrs = ('upstream_name',)

    def __init__(self, name, upstream, builderNames, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(name, builderNames, **kwargs)
        if not interfaces.IScheduler.providedBy(upstream):
            config.error('upstream must be another Scheduler instance')
        self.upstream_name = upstream.name
        self._buildset_new_consumer = None
        self._buildset_complete_consumer = None
        self._cached_upstream_bsids = None
        self._subscription_lock = defer.DeferredLock()

    @defer.inlineCallbacks
    def activate(self):
        if False:
            i = 10
            return i + 15
        yield super().activate()
        if not self.enabled:
            return
        self._buildset_new_consumer = (yield self.master.mq.startConsuming(self._buildset_new_cb, ('buildsets', None, 'new')))
        self._buildset_complete_consumer = (yield self.master.mq.startConsuming(self._buildset_complete_cb, ('buildsets', None, 'complete')))
        yield self._checkCompletedBuildsets(None)

    @defer.inlineCallbacks
    def deactivate(self):
        if False:
            for i in range(10):
                print('nop')
        yield super().deactivate()
        if not self.enabled:
            return
        if self._buildset_new_consumer:
            self._buildset_new_consumer.stopConsuming()
        if self._buildset_complete_consumer:
            self._buildset_complete_consumer.stopConsuming()
        self._cached_upstream_bsids = None

    @util.deferredLocked('_subscription_lock')
    def _buildset_new_cb(self, key, msg):
        if False:
            print('Hello World!')
        if msg['scheduler'] != self.upstream_name:
            return None
        return self._addUpstreamBuildset(msg['bsid'])

    def _buildset_complete_cb(self, key, msg):
        if False:
            while True:
                i = 10
        return self._checkCompletedBuildsets(msg['bsid'])

    @util.deferredLocked('_subscription_lock')
    @defer.inlineCallbacks
    def _checkCompletedBuildsets(self, bsid):
        if False:
            print('Hello World!')
        subs = (yield self._getUpstreamBuildsets())
        sub_bsids = []
        for (sub_bsid, sub_ssids, sub_complete, sub_results) in subs:
            if not sub_complete and sub_bsid != bsid:
                continue
            if sub_results in (SUCCESS, WARNINGS):
                yield self.addBuildsetForSourceStamps(sourcestamps=sub_ssids.copy(), reason='downstream', priority=self.priority)
            sub_bsids.append(sub_bsid)
        yield self._removeUpstreamBuildsets(sub_bsids)

    @defer.inlineCallbacks
    def _updateCachedUpstreamBuilds(self):
        if False:
            for i in range(10):
                print('nop')
        if self._cached_upstream_bsids is None:
            bsids = (yield self.master.db.state.getState(self.objectid, 'upstream_bsids', []))
            self._cached_upstream_bsids = bsids

    @defer.inlineCallbacks
    def _getUpstreamBuildsets(self):
        if False:
            while True:
                i = 10
        yield self._updateCachedUpstreamBuilds()
        changed = False
        rv = []
        for bsid in self._cached_upstream_bsids[:]:
            buildset = (yield self.master.data.get(('buildsets', str(bsid))))
            if not buildset:
                self._cached_upstream_bsids.remove(bsid)
                changed = True
                continue
            ssids = [ss['ssid'] for ss in buildset['sourcestamps']]
            rv.append((bsid, ssids, buildset['complete'], buildset['results']))
        if changed:
            yield self.master.db.state.setState(self.objectid, 'upstream_bsids', self._cached_upstream_bsids)
        return rv

    @defer.inlineCallbacks
    def _addUpstreamBuildset(self, bsid):
        if False:
            while True:
                i = 10
        yield self._updateCachedUpstreamBuilds()
        if bsid not in self._cached_upstream_bsids:
            self._cached_upstream_bsids.append(bsid)
            yield self.master.db.state.setState(self.objectid, 'upstream_bsids', self._cached_upstream_bsids)

    @defer.inlineCallbacks
    def _removeUpstreamBuildsets(self, bsids):
        if False:
            for i in range(10):
                print('nop')
        yield self._updateCachedUpstreamBuilds()
        old = set(self._cached_upstream_bsids)
        self._cached_upstream_bsids = list(old - set(bsids))
        yield self.master.db.state.setState(self.objectid, 'upstream_bsids', self._cached_upstream_bsids)
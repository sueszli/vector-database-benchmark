from twisted.internet import defer
from twisted.internet import task
from buildbot.test.fake import fakemaster

class ChangeSourceMixin:
    """
    This class is used for testing change sources, and handles a few things:

     - starting and stopping a ChangeSource service
     - a fake master with a data API implementation
    """
    changesource = None
    started = False
    DUMMY_CHANGESOURCE_ID = 20
    OTHER_MASTER_ID = 93
    DEFAULT_NAME = 'ChangeSource'

    def setUpChangeSource(self):
        if False:
            while True:
                i = 10
        'Set up the mixin - returns a deferred.'
        self.master = fakemaster.make_master(self, wantDb=True, wantData=True)
        assert not hasattr(self.master, 'addChange')
        return defer.succeed(None)

    @defer.inlineCallbacks
    def tearDownChangeSource(self):
        if False:
            while True:
                i = 10
        'Tear down the mixin - returns a deferred.'
        if not self.started:
            return
        if self.changesource.running:
            yield self.changesource.stopService()
        yield self.changesource.disownServiceParent()
        return

    @defer.inlineCallbacks
    def attachChangeSource(self, cs):
        if False:
            while True:
                i = 10
        'Set up a change source for testing; sets its .master attribute'
        self.changesource = cs
        try:
            self.changesource.master = self.master
        except AttributeError:
            yield self.changesource.setServiceParent(self.master)
        try:
            yield self.changesource.configureService()
        except NotImplementedError:
            pass
        self.changesource.clock = task.Clock()
        return cs

    def startChangeSource(self):
        if False:
            print('Hello World!')
        'start the change source as a service'
        self.started = True
        return self.changesource.startService()

    @defer.inlineCallbacks
    def stopChangeSource(self):
        if False:
            print('Hello World!')
        'stop the change source again; returns a deferred'
        yield self.changesource.stopService()
        self.started = False

    def setChangeSourceToMaster(self, otherMaster):
        if False:
            return 10
        if self.changesource is not None:
            name = self.changesource.name
        else:
            name = self.DEFAULT_NAME
        self.master.data.updates.changesourceIds[name] = self.DUMMY_CHANGESOURCE_ID
        if otherMaster:
            self.master.data.updates.changesourceMasters[self.DUMMY_CHANGESOURCE_ID] = otherMaster
        else:
            del self.master.data.updates.changesourceMasters[self.DUMMY_CHANGESOURCE_ID]
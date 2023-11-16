from twisted.internet import defer
from twisted.python import log
from buildbot.statistics.storage_backends.base import StatsStorageBase
from buildbot.util import service

class StatsService(service.BuildbotService):
    """
    A middleware for passing on statistics data to all storage backends.
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.consumers = []

    def checkConfig(self, storage_backends):
        if False:
            while True:
                i = 10
        for wfb in storage_backends:
            if not isinstance(wfb, StatsStorageBase):
                raise TypeError(f'Invalid type of stats storage service {type(StatsStorageBase)!r}. Should be of type StatsStorageBase, is: {type(StatsStorageBase)!r}')

    @defer.inlineCallbacks
    def reconfigService(self, storage_backends):
        if False:
            while True:
                i = 10
        log.msg(f'Reconfiguring StatsService with config: {storage_backends!r}')
        self.checkConfig(storage_backends)
        self.registeredStorageServices = []
        for svc in storage_backends:
            self.registeredStorageServices.append(svc)
        yield self.removeConsumers()
        yield self.registerConsumers()

    @defer.inlineCallbacks
    def registerConsumers(self):
        if False:
            while True:
                i = 10
        self.consumers = []
        for svc in self.registeredStorageServices:
            for cap in svc.captures:
                cap.parent_svcs.append(svc)
                cap.master = self.master
                consumer = (yield self.master.mq.startConsuming(cap.consume, cap.routingKey))
                self.consumers.append(consumer)

    @defer.inlineCallbacks
    def stopService(self):
        if False:
            return 10
        yield super().stopService()
        yield self.removeConsumers()

    @defer.inlineCallbacks
    def removeConsumers(self):
        if False:
            print('Hello World!')
        for consumer in self.consumers:
            yield consumer.stopConsuming()
        self.consumers = []

    @defer.inlineCallbacks
    def yieldMetricsValue(self, data_name, post_data, buildid):
        if False:
            return 10
        "\n        A method to allow posting data that is not generated and stored as build-data in\n        the database. This method generates the `stats-yield-data` event to the mq layer\n        which is then consumed in self.postData.\n\n        @params\n        data_name: (str) The unique name for identifying this data.\n        post_data: (dict) A dictionary of key-value pairs that'll be sent for storage.\n        buildid: The buildid of the current Build.\n        "
        build_data = (yield self.master.data.get(('builds', buildid)))
        routingKey = ('stats-yieldMetricsValue', 'stats-yield-data')
        msg = {'data_name': data_name, 'post_data': post_data, 'build_data': build_data}
        self.master.mq.produce(routingKey, msg)
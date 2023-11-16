from twisted.internet import defer
from buildbot.process import metrics
from buildbot.util.service import BuildbotServiceManager

class MeasuredBuildbotServiceManager(BuildbotServiceManager):
    managed_services_name = 'services'

    @defer.inlineCallbacks
    def reconfigServiceWithBuildbotConfig(self, new_config):
        if False:
            while True:
                i = 10
        timer = metrics.Timer(f'{self.name}.reconfigServiceWithBuildbotConfig')
        timer.start()
        yield super().reconfigServiceWithBuildbotConfig(new_config)
        metrics.MetricCountEvent.log(f'num_{self.managed_services_name}', len(list(self)), absolute=True)
        timer.stop()
from buildbot.process import buildstep
from buildbot.process.results import SUCCESS
from buildbot.statistics import capture
from buildbot.statistics.storage_backends.base import StatsStorageBase

class FakeStatsStorageService(StatsStorageBase):
    """
    Fake Storage service used in unit tests
    """

    def __init__(self, stats=None, name='FakeStatsStorageService'):
        if False:
            print('Hello World!')
        self.stored_data = []
        if not stats:
            self.stats = [capture.CaptureProperty('TestBuilder', 'test')]
        else:
            self.stats = stats
        self.name = name
        self.captures = []

    def thd_postStatsValue(self, post_data, series_name, context=None):
        if False:
            while True:
                i = 10
        if not context:
            context = {}
        self.stored_data.append((post_data, series_name, context))

class FakeBuildStep(buildstep.BuildStep):
    """
    A fake build step to be used for testing.
    """

    def doSomething(self):
        if False:
            print('Hello World!')
        self.setProperty('test', 10, 'test')

    def start(self):
        if False:
            print('Hello World!')
        self.doSomething()
        return SUCCESS

class FakeInfluxDBClient:
    """
    Fake Influx module for testing on systems that don't have influxdb installed.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.points = []

    def write_points(self, points):
        if False:
            while True:
                i = 10
        self.points.extend(points)
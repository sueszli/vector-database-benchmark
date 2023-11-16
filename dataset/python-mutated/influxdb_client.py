from twisted.python import log
from buildbot import config
from buildbot.statistics.storage_backends.base import StatsStorageBase
try:
    from influxdb import InfluxDBClient
except ImportError:
    InfluxDBClient = None

class InfluxStorageService(StatsStorageBase):
    """
    Delegates data to InfluxDB
    """

    def __init__(self, url, port, user, password, db, captures, name='InfluxStorageService'):
        if False:
            i = 10
            return i + 15
        if not InfluxDBClient:
            config.error('Python client for InfluxDB not installed.')
            return
        self.url = url
        self.port = port
        self.user = user
        self.password = password
        self.db = db
        self.name = name
        self.captures = captures
        self.client = InfluxDBClient(self.url, self.port, self.user, self.password, self.db)
        self._inited = True

    def thd_postStatsValue(self, post_data, series_name, context=None):
        if False:
            i = 10
            return i + 15
        if not self._inited:
            log.err(f'Service {self.name} not initialized')
            return
        data = {'measurement': series_name, 'fields': post_data}
        log.msg('Sending data to InfluxDB')
        log.msg(f'post_data: {post_data!r}')
        if context:
            log.msg(f'context: {context!r}')
            data['tags'] = context
        self.client.write_points([data])
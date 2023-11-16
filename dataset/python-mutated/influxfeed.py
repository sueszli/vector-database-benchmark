from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt
import backtrader.feed as feed
from ..utils import date2num
import datetime as dt
TIMEFRAMES = dict(((bt.TimeFrame.Seconds, 's'), (bt.TimeFrame.Minutes, 'm'), (bt.TimeFrame.Days, 'd'), (bt.TimeFrame.Weeks, 'w'), (bt.TimeFrame.Months, 'm'), (bt.TimeFrame.Years, 'y')))

class InfluxDB(feed.DataBase):
    frompackages = (('influxdb', [('InfluxDBClient', 'idbclient')]), ('influxdb.exceptions', 'InfluxDBClientError'))
    params = (('host', '127.0.0.1'), ('port', '8086'), ('username', None), ('password', None), ('database', None), ('timeframe', bt.TimeFrame.Days), ('startdate', None), ('high', 'high_p'), ('low', 'low_p'), ('open', 'open_p'), ('close', 'close_p'), ('volume', 'volume'), ('ointerest', 'oi'))

    def start(self):
        if False:
            while True:
                i = 10
        super(InfluxDB, self).start()
        try:
            self.ndb = idbclient(self.p.host, self.p.port, self.p.username, self.p.password, self.p.database)
        except InfluxDBClientError as err:
            print('Failed to establish connection to InfluxDB: %s' % err)
        tf = '{multiple}{timeframe}'.format(multiple=self.p.compression if self.p.compression else 1, timeframe=TIMEFRAMES.get(self.p.timeframe, 'd'))
        if not self.p.startdate:
            st = '<= now()'
        else:
            st = ">= '%s'" % self.p.startdate
        qstr = 'SELECT mean("{open_f}") AS "open", mean("{high_f}") AS "high", mean("{low_f}") AS "low", mean("{close_f}") AS "close", mean("{vol_f}") AS "volume", mean("{oi_f}") AS "openinterest" FROM "{dataname}" WHERE time {begin} GROUP BY time({timeframe}) fill(none)'.format(open_f=self.p.open, high_f=self.p.high, low_f=self.p.low, close_f=self.p.close, vol_f=self.p.volume, oi_f=self.p.ointerest, timeframe=tf, begin=st, dataname=self.p.dataname)
        try:
            dbars = list(self.ndb.query(qstr).get_points())
        except InfluxDBClientError as err:
            print('InfluxDB query failed: %s' % err)
        self.biter = iter(dbars)

    def _load(self):
        if False:
            while True:
                i = 10
        try:
            bar = next(self.biter)
        except StopIteration:
            return False
        self.l.datetime[0] = date2num(dt.datetime.strptime(bar['time'], '%Y-%m-%dT%H:%M:%SZ'))
        self.l.open[0] = bar['open']
        self.l.high[0] = bar['high']
        self.l.low[0] = bar['low']
        self.l.close[0] = bar['close']
        self.l.volume[0] = bar['volume']
        return True
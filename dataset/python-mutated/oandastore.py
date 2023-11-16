from __future__ import absolute_import, division, print_function, unicode_literals
import collections
from datetime import datetime, timedelta
import time as _time
import json
import threading
import oandapy
import requests
import backtrader as bt
from backtrader.metabase import MetaParams
from backtrader.utils.py3 import queue, with_metaclass
from backtrader.utils import AutoDict

class OandaRequestError(oandapy.OandaError):

    def __init__(self):
        if False:
            print('Hello World!')
        er = dict(code=599, message='Request Error', description='')
        super(self.__class__, self).__init__(er)

class OandaStreamError(oandapy.OandaError):

    def __init__(self, content=''):
        if False:
            i = 10
            return i + 15
        er = dict(code=598, message='Failed Streaming', description=content)
        super(self.__class__, self).__init__(er)

class OandaTimeFrameError(oandapy.OandaError):

    def __init__(self, content):
        if False:
            for i in range(10):
                print('nop')
        er = dict(code=597, message='Not supported TimeFrame', description='')
        super(self.__class__, self).__init__(er)

class OandaNetworkError(oandapy.OandaError):

    def __init__(self):
        if False:
            return 10
        er = dict(code=596, message='Network Error', description='')
        super(self.__class__, self).__init__(er)

class API(oandapy.API):

    def request(self, endpoint, method='GET', params=None):
        if False:
            print('Hello World!')
        url = '%s/%s' % (self.api_url, endpoint)
        method = method.lower()
        params = params or {}
        func = getattr(self.client, method)
        request_args = {}
        if method == 'get':
            request_args['params'] = params
        else:
            request_args['data'] = params
        try:
            response = func(url, **request_args)
        except requests.RequestException as e:
            return OandaRequestError().error_response
        content = response.content.decode('utf-8')
        content = json.loads(content)
        if response.status_code >= 400:
            return oandapy.OandaError(content).error_response
        return content

class Streamer(oandapy.Streamer):

    def __init__(self, q, headers=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(Streamer, self).__init__(*args, **kwargs)
        if headers:
            self.client.headers.update(headers)
        self.q = q

    def run(self, endpoint, params=None):
        if False:
            print('Hello World!')
        self.connected = True
        params = params or {}
        ignore_heartbeat = None
        if 'ignore_heartbeat' in params:
            ignore_heartbeat = params['ignore_heartbeat']
        request_args = {}
        request_args['params'] = params
        url = '%s/%s' % (self.api_url, endpoint)
        while self.connected:
            try:
                response = self.client.get(url, **request_args)
            except requests.RequestException as e:
                self.q.put(OandaRequestError().error_response)
                break
            if response.status_code != 200:
                self.on_error(response.content)
                break
            try:
                for line in response.iter_lines(chunk_size=None):
                    if not self.connected:
                        break
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if not (ignore_heartbeat and 'heartbeat' in data):
                            self.on_success(data)
            except:
                self.q.put(OandaStreamError().error_response)
                break

    def on_success(self, data):
        if False:
            print('Hello World!')
        if 'tick' in data:
            self.q.put(data['tick'])
        elif 'transaction' in data:
            self.q.put(data['transaction'])

    def on_error(self, data):
        if False:
            return 10
        self.disconnect()
        self.q.put(OandaStreamError(data).error_response)

class MetaSingleton(MetaParams):
    """Metaclass to make a metaclassed class a singleton"""

    def __init__(cls, name, bases, dct):
        if False:
            print('Hello World!')
        super(MetaSingleton, cls).__init__(name, bases, dct)
        cls._singleton = None

    def __call__(cls, *args, **kwargs):
        if False:
            return 10
        if cls._singleton is None:
            cls._singleton = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._singleton

class OandaStore(with_metaclass(MetaSingleton, object)):
    """Singleton class wrapping to control the connections to Oanda.

    Params:

      - ``token`` (default:``None``): API access token

      - ``account`` (default: ``None``): account id

      - ``practice`` (default: ``False``): use the test environment

      - ``account_tmout`` (default: ``10.0``): refresh period for account
        value/cash refresh
    """
    BrokerCls = None
    DataCls = None
    params = (('token', ''), ('account', ''), ('practice', False), ('account_tmout', 10.0))
    _DTEPOCH = datetime(1970, 1, 1)
    _ENVPRACTICE = 'practice'
    _ENVLIVE = 'live'

    @classmethod
    def getdata(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Returns ``DataCls`` with args, kwargs'
        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Returns broker with *args, **kwargs from registered ``BrokerCls``'
        return cls.BrokerCls(*args, **kwargs)

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(OandaStore, self).__init__()
        self.notifs = collections.deque()
        self._env = None
        self.broker = None
        self.datas = list()
        self._orders = collections.OrderedDict()
        self._ordersrev = collections.OrderedDict()
        self._transpend = collections.defaultdict(collections.deque)
        self._oenv = self._ENVPRACTICE if self.p.practice else self._ENVLIVE
        self.oapi = API(environment=self._oenv, access_token=self.p.token, headers={'X-Accept-Datetime-Format': 'UNIX'})
        self._cash = 0.0
        self._value = 0.0
        self._evt_acct = threading.Event()

    def start(self, data=None, broker=None):
        if False:
            i = 10
            return i + 15
        if data is None and broker is None:
            self.cash = None
            return
        if data is not None:
            self._env = data._env
            self.datas.append(data)
            if self.broker is not None:
                self.broker.data_started(data)
        elif broker is not None:
            self.broker = broker
            self.streaming_events()
            self.broker_threads()

    def stop(self):
        if False:
            print('Hello World!')
        if self.broker is not None:
            self.q_ordercreate.put(None)
            self.q_orderclose.put(None)
            self.q_account.put(None)

    def put_notification(self, msg, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.notifs.append((msg, args, kwargs))

    def get_notifications(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the pending "store" notifications'
        self.notifs.append(None)
        return [x for x in iter(self.notifs.popleft, None)]
    _GRANULARITIES = {(bt.TimeFrame.Seconds, 5): 'S5', (bt.TimeFrame.Seconds, 10): 'S10', (bt.TimeFrame.Seconds, 15): 'S15', (bt.TimeFrame.Seconds, 30): 'S30', (bt.TimeFrame.Minutes, 1): 'M1', (bt.TimeFrame.Minutes, 2): 'M3', (bt.TimeFrame.Minutes, 3): 'M3', (bt.TimeFrame.Minutes, 4): 'M4', (bt.TimeFrame.Minutes, 5): 'M5', (bt.TimeFrame.Minutes, 10): 'M5', (bt.TimeFrame.Minutes, 15): 'M5', (bt.TimeFrame.Minutes, 30): 'M5', (bt.TimeFrame.Minutes, 60): 'H1', (bt.TimeFrame.Minutes, 120): 'H2', (bt.TimeFrame.Minutes, 180): 'H3', (bt.TimeFrame.Minutes, 240): 'H4', (bt.TimeFrame.Minutes, 360): 'H6', (bt.TimeFrame.Minutes, 480): 'H8', (bt.TimeFrame.Days, 1): 'D', (bt.TimeFrame.Weeks, 1): 'W', (bt.TimeFrame.Months, 1): 'M'}

    def get_positions(self):
        if False:
            print('Hello World!')
        try:
            positions = self.oapi.get_positions(self.p.account)
        except (oandapy.OandaError, OandaRequestError):
            return None
        poslist = positions.get('positions', [])
        return poslist

    def get_granularity(self, timeframe, compression):
        if False:
            i = 10
            return i + 15
        return self._GRANULARITIES.get((timeframe, compression), None)

    def get_instrument(self, dataname):
        if False:
            while True:
                i = 10
        try:
            insts = self.oapi.get_instruments(self.p.account, instruments=dataname)
        except (oandapy.OandaError, OandaRequestError):
            return None
        i = insts.get('instruments', [{}])
        return i[0] or None

    def streaming_events(self, tmout=None):
        if False:
            while True:
                i = 10
        q = queue.Queue()
        kwargs = {'q': q, 'tmout': tmout}
        t = threading.Thread(target=self._t_streaming_listener, kwargs=kwargs)
        t.daemon = True
        t.start()
        t = threading.Thread(target=self._t_streaming_events, kwargs=kwargs)
        t.daemon = True
        t.start()
        return q

    def _t_streaming_listener(self, q, tmout=None):
        if False:
            i = 10
            return i + 15
        while True:
            trans = q.get()
            self._transaction(trans)

    def _t_streaming_events(self, q, tmout=None):
        if False:
            while True:
                i = 10
        if tmout is not None:
            _time.sleep(tmout)
        streamer = Streamer(q, environment=self._oenv, access_token=self.p.token, headers={'X-Accept-Datetime-Format': 'UNIX'})
        streamer.events(ignore_heartbeat=False)

    def candles(self, dataname, dtbegin, dtend, timeframe, compression, candleFormat, includeFirst):
        if False:
            print('Hello World!')
        kwargs = locals().copy()
        kwargs.pop('self')
        kwargs['q'] = q = queue.Queue()
        t = threading.Thread(target=self._t_candles, kwargs=kwargs)
        t.daemon = True
        t.start()
        return q

    def _t_candles(self, dataname, dtbegin, dtend, timeframe, compression, candleFormat, includeFirst, q):
        if False:
            print('Hello World!')
        granularity = self.get_granularity(timeframe, compression)
        if granularity is None:
            e = OandaTimeFrameError()
            q.put(e.error_response)
            return
        dtkwargs = {}
        if dtbegin is not None:
            dtkwargs['start'] = int((dtbegin - self._DTEPOCH).total_seconds())
        if dtend is not None:
            dtkwargs['end'] = int((dtend - self._DTEPOCH).total_seconds())
        try:
            response = self.oapi.get_history(instrument=dataname, granularity=granularity, candleFormat=candleFormat, **dtkwargs)
        except oandapy.OandaError as e:
            q.put(e.error_response)
            q.put(None)
            return
        for candle in response.get('candles', []):
            q.put(candle)
        q.put({})

    def streaming_prices(self, dataname, tmout=None):
        if False:
            print('Hello World!')
        q = queue.Queue()
        kwargs = {'q': q, 'dataname': dataname, 'tmout': tmout}
        t = threading.Thread(target=self._t_streaming_prices, kwargs=kwargs)
        t.daemon = True
        t.start()
        return q

    def _t_streaming_prices(self, dataname, q, tmout):
        if False:
            print('Hello World!')
        if tmout is not None:
            _time.sleep(tmout)
        streamer = Streamer(q, environment=self._oenv, access_token=self.p.token, headers={'X-Accept-Datetime-Format': 'UNIX'})
        streamer.rates(self.p.account, instruments=dataname)

    def get_cash(self):
        if False:
            return 10
        return self._cash

    def get_value(self):
        if False:
            print('Hello World!')
        return self._value
    _ORDEREXECS = {bt.Order.Market: 'market', bt.Order.Limit: 'limit', bt.Order.Stop: 'stop', bt.Order.StopLimit: 'stop'}

    def broker_threads(self):
        if False:
            for i in range(10):
                print('nop')
        self.q_account = queue.Queue()
        self.q_account.put(True)
        t = threading.Thread(target=self._t_account)
        t.daemon = True
        t.start()
        self.q_ordercreate = queue.Queue()
        t = threading.Thread(target=self._t_order_create)
        t.daemon = True
        t.start()
        self.q_orderclose = queue.Queue()
        t = threading.Thread(target=self._t_order_cancel)
        t.daemon = True
        t.start()
        self._evt_acct.wait(self.p.account_tmout)

    def _t_account(self):
        if False:
            return 10
        while True:
            try:
                msg = self.q_account.get(timeout=self.p.account_tmout)
                if msg is None:
                    break
            except queue.Empty:
                pass
            try:
                accinfo = self.oapi.get_account(self.p.account)
            except Exception as e:
                self.put_notification(e)
                continue
            try:
                self._cash = accinfo['marginAvail']
                self._value = accinfo['balance']
            except KeyError:
                pass
            self._evt_acct.set()

    def order_create(self, order, stopside=None, takeside=None, **kwargs):
        if False:
            return 10
        okwargs = dict()
        okwargs['instrument'] = order.data._dataname
        okwargs['units'] = abs(order.created.size)
        okwargs['side'] = 'buy' if order.isbuy() else 'sell'
        okwargs['type'] = self._ORDEREXECS[order.exectype]
        if order.exectype != bt.Order.Market:
            okwargs['price'] = order.created.price
            if order.valid is None:
                valid = datetime.utcnow() + timedelta(days=30)
            else:
                valid = order.data.num2date(order.valid)
            okwargs['expiry'] = int((valid - self._DTEPOCH).total_seconds())
        if order.exectype == bt.Order.StopLimit:
            okwargs['lowerBound'] = order.created.pricelimit
            okwargs['upperBound'] = order.created.pricelimit
        if order.exectype == bt.Order.StopTrail:
            okwargs['trailingStop'] = order.trailamount
        if stopside is not None:
            okwargs['stopLoss'] = stopside.price
        if takeside is not None:
            okwargs['takeProfit'] = takeside.price
        okwargs.update(**kwargs)
        self.q_ordercreate.put((order.ref, okwargs))
        return order
    _OIDSINGLE = ['orderOpened', 'tradeOpened', 'tradeReduced']
    _OIDMULTIPLE = ['tradesClosed']

    def _t_order_create(self):
        if False:
            return 10
        while True:
            msg = self.q_ordercreate.get()
            if msg is None:
                break
            (oref, okwargs) = msg
            try:
                o = self.oapi.create_order(self.p.account, **okwargs)
            except Exception as e:
                self.put_notification(e)
                self.broker._reject(oref)
                return
            oids = list()
            for oidfield in self._OIDSINGLE:
                if oidfield in o and 'id' in o[oidfield]:
                    oids.append(o[oidfield]['id'])
            for oidfield in self._OIDMULTIPLE:
                if oidfield in o:
                    for suboidfield in o[oidfield]:
                        oids.append(suboidfield['id'])
            if not oids:
                self.broker._reject(oref)
                return
            self._orders[oref] = oids[0]
            self.broker._submit(oref)
            if okwargs['type'] == 'market':
                self.broker._accept(oref)
            for oid in oids:
                self._ordersrev[oid] = oref
                tpending = self._transpend[oid]
                tpending.append(None)
                while True:
                    trans = tpending.popleft()
                    if trans is None:
                        break
                    self._process_transaction(oid, trans)

    def order_cancel(self, order):
        if False:
            i = 10
            return i + 15
        self.q_orderclose.put(order.ref)
        return order

    def _t_order_cancel(self):
        if False:
            i = 10
            return i + 15
        while True:
            oref = self.q_orderclose.get()
            if oref is None:
                break
            oid = self._orders.get(oref, None)
            if oid is None:
                continue
            try:
                o = self.oapi.close_order(self.p.account, oid)
            except Exception as e:
                continue
            self.broker._cancel(oref)
    _X_ORDER_CREATE = ('STOP_ORDER_CREATE', 'LIMIT_ORDER_CREATE', 'MARKET_IF_TOUCHED_ORDER_CREATE')

    def _transaction(self, trans):
        if False:
            while True:
                i = 10
        ttype = trans['type']
        if ttype == 'MARKET_ORDER_CREATE':
            try:
                oid = trans['tradeReduced']['id']
            except KeyError:
                try:
                    oid = trans['tradeOpened']['id']
                except KeyError:
                    return
        elif ttype in self._X_ORDER_CREATE:
            oid = trans['id']
        elif ttype == 'ORDER_FILLED':
            oid = trans['orderId']
        elif ttype == 'ORDER_CANCEL':
            oid = trans['orderId']
        elif ttype == 'TRADE_CLOSE':
            oid = trans['id']
            pid = trans['tradeId']
            if pid in self._orders and False:
                return
            msg = 'Received TRADE_CLOSE for unknown order, possibly generated over a different client or GUI'
            self.put_notification(msg, trans)
            return
        else:
            try:
                oid = trans['id']
            except KeyError:
                oid = 'None'
            msg = 'Received {} with oid {}. Unknown situation'
            msg = msg.format(ttype, oid)
            self.put_notification(msg, trans)
            return
        try:
            oref = self._ordersrev[oid]
            self._process_transaction(oid, trans)
        except KeyError:
            self._transpend[oid].append(trans)
    _X_ORDER_FILLED = ('MARKET_ORDER_CREATE', 'ORDER_FILLED', 'TAKE_PROFIT_FILLED', 'STOP_LOSS_FILLED', 'TRAILING_STOP_FILLED')

    def _process_transaction(self, oid, trans):
        if False:
            for i in range(10):
                print('nop')
        try:
            oref = self._ordersrev.pop(oid)
        except KeyError:
            return
        ttype = trans['type']
        if ttype in self._X_ORDER_FILLED:
            size = trans['units']
            if trans['side'] == 'sell':
                size = -size
            price = trans['price']
            self.broker._fill(oref, size, price, ttype=ttype)
        elif ttype in self._X_ORDER_CREATE:
            self.broker._accept(oref)
            self._ordersrev[oid] = oref
        elif ttype in 'ORDER_CANCEL':
            reason = trans['reason']
            if reason == 'ORDER_FILLED':
                pass
            elif reason == 'TIME_IN_FORCE_EXPIRED':
                self.broker._expire(oref)
            elif reason == 'CLIENT_REQUEST':
                self.broker._cancel(oref)
            else:
                self.broker._reject(oref)
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
from copy import copy
from datetime import date, datetime, timedelta
import threading
import uuid
import ib.ext.Order
import ib.opt as ibopt
from backtrader.feed import DataBase
from backtrader import TimeFrame, num2date, date2num, BrokerBase, Order, OrderBase, OrderData
from backtrader.utils.py3 import bytes, bstr, with_metaclass, queue, MAXFLOAT
from backtrader.metabase import MetaParams
from backtrader.comminfo import CommInfoBase
from backtrader.position import Position
from backtrader.stores import ibstore
from backtrader.utils import AutoDict, AutoOrderedDict
from backtrader.comminfo import CommInfoBase
bytes = bstr

class IBOrderState(object):
    _fields = ['status', 'initMargin', 'maintMargin', 'equityWithLoan', 'commission', 'minCommission', 'maxCommission', 'commissionCurrency', 'warningText']

    def __init__(self, orderstate):
        if False:
            return 10
        for f in self._fields:
            fname = 'm_' + f
            setattr(self, fname, getattr(orderstate, fname))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        txt = list()
        txt.append('--- ORDERSTATE BEGIN')
        for f in self._fields:
            fname = 'm_' + f
            txt.append('{}: {}'.format(f.capitalize(), getattr(self, fname)))
        txt.append('--- ORDERSTATE END')
        return '\n'.join(txt)

class IBOrder(OrderBase, ib.ext.Order.Order):
    """Subclasses the IBPy order to provide the minimum extra functionality
    needed to be compatible with the internally defined orders

    Once ``OrderBase`` has processed the parameters, the __init__ method takes
    over to use the parameter values and set the appropriate values in the
    ib.ext.Order.Order object

    Any extra parameters supplied with kwargs are applied directly to the
    ib.ext.Order.Order object, which could be used as follows::

      Example: if the 4 order execution types directly supported by
      ``backtrader`` are not enough, in the case of for example
      *Interactive Brokers* the following could be passed as *kwargs*::

        orderType='LIT', lmtPrice=10.0, auxPrice=9.8

      This would override the settings created by ``backtrader`` and
      generate a ``LIMIT IF TOUCHED`` order with a *touched* price of 9.8
      and a *limit* price of 10.0.

    This would be done almost always from the ``Buy`` and ``Sell`` methods of
    the ``Strategy`` subclass being used in ``Cerebro``
    """

    def __str__(self):
        if False:
            while True:
                i = 10
        'Get the printout from the base class and add some ib.Order specific\n        fields'
        basetxt = super(IBOrder, self).__str__()
        tojoin = [basetxt]
        tojoin.append('Ref: {}'.format(self.ref))
        tojoin.append('orderId: {}'.format(self.m_orderId))
        tojoin.append('Action: {}'.format(self.m_action))
        tojoin.append('Size (ib): {}'.format(self.m_totalQuantity))
        tojoin.append('Lmt Price: {}'.format(self.m_lmtPrice))
        tojoin.append('Aux Price: {}'.format(self.m_auxPrice))
        tojoin.append('OrderType: {}'.format(self.m_orderType))
        tojoin.append('Tif (Time in Force): {}'.format(self.m_tif))
        tojoin.append('GoodTillDate: {}'.format(self.m_goodTillDate))
        return '\n'.join(tojoin)
    _IBOrdTypes = {None: bytes('MKT'), Order.Market: bytes('MKT'), Order.Limit: bytes('LMT'), Order.Close: bytes('MOC'), Order.Stop: bytes('STP'), Order.StopLimit: bytes('STPLMT'), Order.StopTrail: bytes('TRAIL'), Order.StopTrailLimit: bytes('TRAIL LIMIT')}

    def __init__(self, action, **kwargs):
        if False:
            while True:
                i = 10
        self._willexpire = False
        self.ordtype = self.Buy if action == 'BUY' else self.Sell
        super(IBOrder, self).__init__()
        ib.ext.Order.Order.__init__(self)
        self.m_orderType = self._IBOrdTypes[self.exectype]
        self.m_permid = 0
        self.m_action = bytes(action)
        self.m_lmtPrice = 0.0
        self.m_auxPrice = 0.0
        if self.exectype == self.Market:
            pass
        elif self.exectype == self.Close:
            pass
        elif self.exectype == self.Limit:
            self.m_lmtPrice = self.price
        elif self.exectype == self.Stop:
            self.m_auxPrice = self.price
        elif self.exectype == self.StopLimit:
            self.m_lmtPrice = self.pricelimit
            self.m_auxPrice = self.price
        elif self.exectype == self.StopTrail:
            if self.trailamount is not None:
                self.m_auxPrice = self.trailamount
            elif self.trailpercent is not None:
                self.m_trailingPercent = self.trailpercent * 100.0
        elif self.exectype == self.StopTrailLimit:
            self.m_trailStopPrice = self.m_lmtPrice = self.price
            self.m_lmtPrice = self.pricelimit
            if self.trailamount is not None:
                self.m_auxPrice = self.trailamount
            elif self.trailpercent is not None:
                self.m_trailingPercent = self.trailpercent * 100.0
        self.m_totalQuantity = abs(self.size)
        self.m_transmit = self.transmit
        if self.parent is not None:
            self.m_parentId = self.parent.m_orderId
        if self.valid is None:
            tif = 'GTC'
        elif isinstance(self.valid, (datetime, date)):
            tif = 'GTD'
            self.m_goodTillDate = bytes(self.valid.strftime('%Y%m%d %H:%M:%S'))
        elif isinstance(self.valid, (timedelta,)):
            if self.valid == self.DAY:
                tif = 'DAY'
            else:
                tif = 'GTD'
                valid = datetime.now() + self.valid
                self.m_goodTillDate = bytes(valid.strftime('%Y%m%d %H:%M:%S'))
        elif self.valid == 0:
            tif = 'DAY'
        else:
            tif = 'GTD'
            valid = num2date(self.valid)
            self.m_goodTillDate = bytes(valid.strftime('%Y%m%d %H:%M:%S'))
        self.m_tif = bytes(tif)
        self.m_ocaType = 1
        for k in kwargs:
            setattr(self, (not hasattr(self, k)) * 'm_' + k, kwargs[k])

class IBCommInfo(CommInfoBase):
    """
    Commissions are calculated by ib, but the trades calculations in the
    ```Strategy`` rely on the order carrying a CommInfo object attached for the
    calculation of the operation cost and value.

    These are non-critical informations, but removing them from the trade could
    break existing usage and it is better to provide a CommInfo objet which
    enables those calculations even if with approvimate values.

    The margin calculation is not a known in advance information with IB
    (margin impact can be gotten from OrderState objects) and therefore it is
    left as future exercise to get it"""

    def getvaluesize(self, size, price):
        if False:
            return 10
        return abs(size) * price

    def getoperationcost(self, size, price):
        if False:
            while True:
                i = 10
        'Returns the needed amount of cash an operation would cost'
        return abs(size) * price

class MetaIBBroker(BrokerBase.__class__):

    def __init__(cls, name, bases, dct):
        if False:
            for i in range(10):
                print('nop')
        'Class has already been created ... register'
        super(MetaIBBroker, cls).__init__(name, bases, dct)
        ibstore.IBStore.BrokerCls = cls

class IBBroker(with_metaclass(MetaIBBroker, BrokerBase)):
    """Broker implementation for Interactive Brokers.

    This class maps the orders/positions from Interactive Brokers to the
    internal API of ``backtrader``.

    Notes:

      - ``tradeid`` is not really supported, because the profit and loss are
        taken directly from IB. Because (as expected) calculates it in FIFO
        manner, the pnl is not accurate for the tradeid.

      - Position

        If there is an open position for an asset at the beginning of
        operaitons or orders given by other means change a position, the trades
        calculated in the ``Strategy`` in cerebro will not reflect the reality.

        To avoid this, this broker would have to do its own position
        management which would also allow tradeid with multiple ids (profit and
        loss would also be calculated locally), but could be considered to be
        defeating the purpose of working with a live broker
    """
    params = ()

    def __init__(self, **kwargs):
        if False:
            return 10
        super(IBBroker, self).__init__()
        self.ib = ibstore.IBStore(**kwargs)
        self.startingcash = self.cash = 0.0
        self.startingvalue = self.value = 0.0
        self._lock_orders = threading.Lock()
        self.orderbyid = dict()
        self.executions = dict()
        self.ordstatus = collections.defaultdict(dict)
        self.notifs = queue.Queue()
        self.tonotify = collections.deque()

    def start(self):
        if False:
            i = 10
            return i + 15
        super(IBBroker, self).start()
        self.ib.start(broker=self)
        if self.ib.connected():
            self.ib.reqAccountUpdates()
            self.startingcash = self.cash = self.ib.get_acc_cash()
            self.startingvalue = self.value = self.ib.get_acc_value()
        else:
            self.startingcash = self.cash = 0.0
            self.startingvalue = self.value = 0.0

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        super(IBBroker, self).stop()
        self.ib.stop()

    def getcash(self):
        if False:
            print('Hello World!')
        self.cash = self.ib.get_acc_cash()
        return self.cash

    def getvalue(self, datas=None):
        if False:
            print('Hello World!')
        self.value = self.ib.get_acc_value()
        return self.value

    def getposition(self, data, clone=True):
        if False:
            for i in range(10):
                print('nop')
        return self.ib.getposition(data.tradecontract, clone=clone)

    def cancel(self, order):
        if False:
            for i in range(10):
                print('nop')
        try:
            o = self.orderbyid[order.m_orderId]
        except (ValueError, KeyError):
            return
        if order.status == Order.Cancelled:
            return
        self.ib.cancelOrder(order.m_orderId)

    def orderstatus(self, order):
        if False:
            print('Hello World!')
        try:
            o = self.orderbyid[order.m_orderId]
        except (ValueError, KeyError):
            o = order
        return o.status

    def submit(self, order):
        if False:
            print('Hello World!')
        order.submit(self)
        if order.oco is None:
            order.m_ocaGroup = bytes(uuid.uuid4())
        else:
            order.m_ocaGroup = self.orderbyid[order.oco.m_orderId].m_ocaGroup
        self.orderbyid[order.m_orderId] = order
        self.ib.placeOrder(order.m_orderId, order.data.tradecontract, order)
        self.notify(order)
        return order

    def getcommissioninfo(self, data):
        if False:
            for i in range(10):
                print('nop')
        contract = data.tradecontract
        try:
            mult = float(contract.m_multiplier)
        except (ValueError, TypeError):
            mult = 1.0
        stocklike = contract.m_secType not in ('FUT', 'OPT', 'FOP')
        return IBCommInfo(mult=mult, stocklike=stocklike)

    def _makeorder(self, action, owner, data, size, price=None, plimit=None, exectype=None, valid=None, tradeid=0, **kwargs):
        if False:
            while True:
                i = 10
        order = IBOrder(action, owner=owner, data=data, size=size, price=price, pricelimit=plimit, exectype=exectype, valid=valid, tradeid=tradeid, m_clientId=self.ib.clientId, m_orderId=self.ib.nextOrderId(), **kwargs)
        order.addcomminfo(self.getcommissioninfo(data))
        return order

    def buy(self, owner, data, size, price=None, plimit=None, exectype=None, valid=None, tradeid=0, **kwargs):
        if False:
            i = 10
            return i + 15
        order = self._makeorder('BUY', owner, data, size, price, plimit, exectype, valid, tradeid, **kwargs)
        return self.submit(order)

    def sell(self, owner, data, size, price=None, plimit=None, exectype=None, valid=None, tradeid=0, **kwargs):
        if False:
            while True:
                i = 10
        order = self._makeorder('SELL', owner, data, size, price, plimit, exectype, valid, tradeid, **kwargs)
        return self.submit(order)

    def notify(self, order):
        if False:
            return 10
        self.notifs.put(order.clone())

    def get_notification(self):
        if False:
            return 10
        try:
            return self.notifs.get(False)
        except queue.Empty:
            pass
        return None

    def next(self):
        if False:
            i = 10
            return i + 15
        self.notifs.put(None)
    (SUBMITTED, FILLED, CANCELLED, INACTIVE, PENDINGSUBMIT, PENDINGCANCEL, PRESUBMITTED) = ('Submitted', 'Filled', 'Cancelled', 'Inactive', 'PendingSubmit', 'PendingCancel', 'PreSubmitted')

    def push_orderstatus(self, msg):
        if False:
            while True:
                i = 10
        try:
            order = self.orderbyid[msg.orderId]
        except KeyError:
            return
        if msg.status == self.SUBMITTED and msg.filled == 0:
            if order.status == order.Accepted:
                return
            order.accept(self)
            self.notify(order)
        elif msg.status == self.CANCELLED:
            if order.status in [order.Cancelled, order.Expired]:
                return
            if order._willexpire:
                order.expire()
            else:
                order.cancel()
            self.notify(order)
        elif msg.status == self.PENDINGCANCEL:
            if order.status == order.Cancelled:
                return
        elif msg.status == self.INACTIVE:
            if order.status == order.Rejected:
                return
            order.reject(self)
            self.notify(order)
        elif msg.status in [self.SUBMITTED, self.FILLED]:
            self.ordstatus[msg.orderId][msg.filled] = msg
        elif msg.status in [self.PENDINGSUBMIT, self.PRESUBMITTED]:
            if msg.filled:
                self.ordstatus[msg.orderId][msg.filled] = msg
        else:
            pass

    def push_execution(self, ex):
        if False:
            while True:
                i = 10
        self.executions[ex.m_execId] = ex

    def push_commissionreport(self, cr):
        if False:
            while True:
                i = 10
        with self._lock_orders:
            ex = self.executions.pop(cr.m_execId)
            oid = ex.m_orderId
            order = self.orderbyid[oid]
            ostatus = self.ordstatus[oid].pop(ex.m_cumQty)
            position = self.getposition(order.data, clone=False)
            pprice_orig = position.price
            size = ex.m_shares if ex.m_side[0] == 'B' else -ex.m_shares
            price = ex.m_price
            (psize, pprice, opened, closed) = position.update(size, price)
            comm = cr.m_commission
            closedcomm = comm * closed / size
            openedcomm = comm - closedcomm
            comminfo = order.comminfo
            closedvalue = comminfo.getoperationcost(closed, pprice_orig)
            openedvalue = comminfo.getoperationcost(opened, price)
            pnl = cr.m_realizedPNL if closed else 0.0
            dt = date2num(datetime.strptime(ex.m_time, '%Y%m%d  %H:%M:%S'))
            margin = order.data.close[0]
            order.execute(dt, size, price, closed, closedvalue, closedcomm, opened, openedvalue, openedcomm, margin, pnl, psize, pprice)
            if ostatus.status == self.FILLED:
                order.completed()
                self.ordstatus.pop(oid)
            else:
                order.partial()
            if oid not in self.tonotify:
                self.tonotify.append(oid)

    def push_portupdate(self):
        if False:
            for i in range(10):
                print('nop')
        with self._lock_orders:
            while self.tonotify:
                oid = self.tonotify.popleft()
                order = self.orderbyid[oid]
                self.notify(order)

    def push_ordererror(self, msg):
        if False:
            while True:
                i = 10
        with self._lock_orders:
            try:
                order = self.orderbyid[msg.id]
            except (KeyError, AttributeError):
                return
            if msg.errorCode == 202:
                if not order.alive():
                    return
                order.cancel()
            elif msg.errorCode == 201:
                if order.status == order.Rejected:
                    return
                order.reject()
            else:
                order.reject()
            self.notify(order)

    def push_orderstate(self, msg):
        if False:
            i = 10
            return i + 15
        with self._lock_orders:
            try:
                order = self.orderbyid[msg.orderId]
            except (KeyError, AttributeError):
                return
            if msg.orderState.m_status in ['PendingCancel', 'Cancelled', 'Canceled']:
                order._willexpire = True
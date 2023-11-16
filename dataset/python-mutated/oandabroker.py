from __future__ import absolute_import, division, print_function, unicode_literals
import collections
from copy import copy
from datetime import date, datetime, timedelta
import threading
from backtrader.feed import DataBase
from backtrader import TimeFrame, num2date, date2num, BrokerBase, Order, BuyOrder, SellOrder, OrderBase, OrderData
from backtrader.utils.py3 import bytes, with_metaclass, MAXFLOAT
from backtrader.metabase import MetaParams
from backtrader.comminfo import CommInfoBase
from backtrader.position import Position
from backtrader.stores import oandastore
from backtrader.utils import AutoDict, AutoOrderedDict
from backtrader.comminfo import CommInfoBase

class OandaCommInfo(CommInfoBase):

    def getvaluesize(self, size, price):
        if False:
            i = 10
            return i + 15
        return abs(size) * price

    def getoperationcost(self, size, price):
        if False:
            while True:
                i = 10
        'Returns the needed amount of cash an operation would cost'
        return abs(size) * price

class MetaOandaBroker(BrokerBase.__class__):

    def __init__(cls, name, bases, dct):
        if False:
            while True:
                i = 10
        'Class has already been created ... register'
        super(MetaOandaBroker, cls).__init__(name, bases, dct)
        oandastore.OandaStore.BrokerCls = cls

class OandaBroker(with_metaclass(MetaOandaBroker, BrokerBase)):
    """Broker implementation for Oanda.

    This class maps the orders/positions from Oanda to the
    internal API of ``backtrader``.

    Params:

      - ``use_positions`` (default:``True``): When connecting to the broker
        provider use the existing positions to kickstart the broker.

        Set to ``False`` during instantiation to disregard any existing
        position
    """
    params = (('use_positions', True), ('commission', OandaCommInfo(mult=1.0, stocklike=False)))

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(OandaBroker, self).__init__()
        self.o = oandastore.OandaStore(**kwargs)
        self.orders = collections.OrderedDict()
        self.notifs = collections.deque()
        self.opending = collections.defaultdict(list)
        self.brackets = dict()
        self.startingcash = self.cash = 0.0
        self.startingvalue = self.value = 0.0
        self.positions = collections.defaultdict(Position)

    def start(self):
        if False:
            while True:
                i = 10
        super(OandaBroker, self).start()
        self.o.start(broker=self)
        self.startingcash = self.cash = cash = self.o.get_cash()
        self.startingvalue = self.value = self.o.get_value()
        if self.p.use_positions:
            for p in self.o.get_positions():
                print('position for instrument:', p['instrument'])
                is_sell = p['side'] == 'sell'
                size = p['units']
                if is_sell:
                    size = -size
                price = p['avgPrice']
                self.positions[p['instrument']] = Position(size, price)

    def data_started(self, data):
        if False:
            i = 10
            return i + 15
        pos = self.getposition(data)
        if pos.size < 0:
            order = SellOrder(data=data, size=pos.size, price=pos.price, exectype=Order.Market, simulated=True)
            order.addcomminfo(self.getcommissioninfo(data))
            order.execute(0, pos.size, pos.price, 0, 0.0, 0.0, pos.size, 0.0, 0.0, 0.0, 0.0, pos.size, pos.price)
            order.completed()
            self.notify(order)
        elif pos.size > 0:
            order = BuyOrder(data=data, size=pos.size, price=pos.price, exectype=Order.Market, simulated=True)
            order.addcomminfo(self.getcommissioninfo(data))
            order.execute(0, pos.size, pos.price, 0, 0.0, 0.0, pos.size, 0.0, 0.0, 0.0, 0.0, pos.size, pos.price)
            order.completed()
            self.notify(order)

    def stop(self):
        if False:
            return 10
        super(OandaBroker, self).stop()
        self.o.stop()

    def getcash(self):
        if False:
            while True:
                i = 10
        self.cash = cash = self.o.get_cash()
        return cash

    def getvalue(self, datas=None):
        if False:
            print('Hello World!')
        self.value = self.o.get_value()
        return self.value

    def getposition(self, data, clone=True):
        if False:
            while True:
                i = 10
        pos = self.positions[data._dataname]
        if clone:
            pos = pos.clone()
        return pos

    def orderstatus(self, order):
        if False:
            print('Hello World!')
        o = self.orders[order.ref]
        return o.status

    def _submit(self, oref):
        if False:
            i = 10
            return i + 15
        order = self.orders[oref]
        order.submit(self)
        self.notify(order)
        for o in self._bracketnotif(order):
            o.submit(self)
            self.notify(o)

    def _reject(self, oref):
        if False:
            for i in range(10):
                print('nop')
        order = self.orders[oref]
        order.reject(self)
        self.notify(order)
        self._bracketize(order, cancel=True)

    def _accept(self, oref):
        if False:
            return 10
        order = self.orders[oref]
        order.accept()
        self.notify(order)
        for o in self._bracketnotif(order):
            o.accept(self)
            self.notify(o)

    def _cancel(self, oref):
        if False:
            for i in range(10):
                print('nop')
        order = self.orders[oref]
        order.cancel()
        self.notify(order)
        self._bracketize(order, cancel=True)

    def _expire(self, oref):
        if False:
            i = 10
            return i + 15
        order = self.orders[oref]
        order.expire()
        self.notify(order)
        self._bracketize(order, cancel=True)

    def _bracketnotif(self, order):
        if False:
            i = 10
            return i + 15
        pref = getattr(order.parent, 'ref', order.ref)
        br = self.brackets.get(pref, None)
        return br[-2:] if br is not None else []

    def _bracketize(self, order, cancel=False):
        if False:
            for i in range(10):
                print('nop')
        pref = getattr(order.parent, 'ref', order.ref)
        br = self.brackets.pop(pref, None)
        if br is None:
            return
        if not cancel:
            if len(br) == 3:
                br = br[1:]
                for o in br:
                    o.activate()
                self.brackets[pref] = br
            elif len(br) == 2:
                oidx = br.index(order)
                self._cancel(br[1 - oidx].ref)
        else:
            for o in br:
                if o.alive():
                    self._cancel(o.ref)

    def _fill(self, oref, size, price, ttype, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        order = self.orders[oref]
        if not order.alive():
            pref = getattr(order.parent, 'ref', order.ref)
            if pref not in self.brackets:
                msg = 'Order fill received for {}, with price {} and size {} but order is no longer alive and is not a bracket. Unknown situation'
                msg.format(order.ref, price, size)
                self.put_notification(msg, order, price, size)
                return
            if ttype == 'STOP_LOSS_FILLED':
                order = self.brackets[pref][-2]
            elif ttype == 'TAKE_PROFIT_FILLED':
                order = self.brackets[pref][-1]
            else:
                msg = 'Order fill received for {}, with price {} and size {} but order is no longer alive and is a bracket. Unknown situation'
                msg.format(order.ref, price, size)
                self.put_notification(msg, order, price, size)
                return
        data = order.data
        pos = self.getposition(data, clone=False)
        (psize, pprice, opened, closed) = pos.update(size, price)
        comminfo = self.getcommissioninfo(data)
        closedvalue = closedcomm = 0.0
        openedvalue = openedcomm = 0.0
        margin = pnl = 0.0
        order.execute(data.datetime[0], size, price, closed, closedvalue, closedcomm, opened, openedvalue, openedcomm, margin, pnl, psize, pprice)
        if order.executed.remsize:
            order.partial()
            self.notify(order)
        else:
            order.completed()
            self.notify(order)
            self._bracketize(order)

    def _transmit(self, order):
        if False:
            i = 10
            return i + 15
        oref = order.ref
        pref = getattr(order.parent, 'ref', oref)
        if order.transmit:
            if oref != pref:
                takeside = order
                (parent, stopside) = self.opending.pop(pref)
                for o in (parent, stopside, takeside):
                    self.orders[o.ref] = o
                self.brackets[pref] = [parent, stopside, takeside]
                self.o.order_create(parent, stopside, takeside)
                return takeside
            else:
                self.orders[order.ref] = order
                return self.o.order_create(order)
        self.opending[pref].append(order)
        return order

    def buy(self, owner, data, size, price=None, plimit=None, exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, trailpercent=None, parent=None, transmit=True, **kwargs):
        if False:
            print('Hello World!')
        order = BuyOrder(owner=owner, data=data, size=size, price=price, pricelimit=plimit, exectype=exectype, valid=valid, tradeid=tradeid, trailamount=trailamount, trailpercent=trailpercent, parent=parent, transmit=transmit)
        order.addinfo(**kwargs)
        order.addcomminfo(self.getcommissioninfo(data))
        return self._transmit(order)

    def sell(self, owner, data, size, price=None, plimit=None, exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, trailpercent=None, parent=None, transmit=True, **kwargs):
        if False:
            while True:
                i = 10
        order = SellOrder(owner=owner, data=data, size=size, price=price, pricelimit=plimit, exectype=exectype, valid=valid, tradeid=tradeid, trailamount=trailamount, trailpercent=trailpercent, parent=parent, transmit=transmit)
        order.addinfo(**kwargs)
        order.addcomminfo(self.getcommissioninfo(data))
        return self._transmit(order)

    def cancel(self, order):
        if False:
            for i in range(10):
                print('nop')
        o = self.orders[order.ref]
        if order.status == Order.Cancelled:
            return
        return self.o.order_cancel(order)

    def notify(self, order):
        if False:
            return 10
        self.notifs.append(order.clone())

    def get_notification(self):
        if False:
            while True:
                i = 10
        if not self.notifs:
            return None
        return self.notifs.popleft()

    def next(self):
        if False:
            while True:
                i = 10
        self.notifs.append(None)
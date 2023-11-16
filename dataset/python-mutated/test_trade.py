from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
from backtrader import trade

class FakeCommInfo(object):

    def getvaluesize(self, size, price):
        if False:
            print('Hello World!')
        return 0

    def profitandloss(self, size, price, newprice):
        if False:
            while True:
                i = 10
        return 0

class FakeData(object):
    """
    Minimal interface to avoid errors when trade tries to get information from
    the data during the test
    """

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return 0

    @property
    def datetime(self):
        if False:
            print('Hello World!')
        return [0.0]

    @property
    def close(self):
        if False:
            i = 10
            return i + 15
        return [0.0]

def test_run(main=False):
    if False:
        for i in range(10):
            print('nop')
    tr = trade.Trade(data=FakeData())
    order = bt.BuyOrder(data=FakeData(), size=0, price=1.0, exectype=bt.Order.Market, simulated=True)
    commrate = 0.025
    size = 10
    price = 10.0
    value = size * price
    commission = value * commrate
    tr.update(order=order, size=size, price=price, value=value, commission=commission, pnl=0.0, comminfo=FakeCommInfo())
    assert not tr.isclosed
    assert tr.size == size
    assert tr.price == price
    assert tr.commission == commission
    assert not tr.pnl
    assert tr.pnlcomm == tr.pnl - tr.commission
    upsize = -5
    upprice = 12.5
    upvalue = upsize * upprice
    upcomm = abs(value) * commrate
    tr.update(order=order, size=upsize, price=upprice, value=upvalue, commission=upcomm, pnl=0.0, comminfo=FakeCommInfo())
    assert not tr.isclosed
    assert tr.size == size + upsize
    assert tr.price == price
    assert tr.commission == commission + upcomm
    size = tr.size
    price = tr.price
    commission = tr.commission
    upsize = 7
    upprice = 14.5
    upvalue = upsize * upprice
    upcomm = abs(value) * commrate
    tr.update(order=order, size=upsize, price=upprice, value=upvalue, commission=upcomm, pnl=0.0, comminfo=FakeCommInfo())
    assert not tr.isclosed
    assert tr.size == size + upsize
    assert tr.price == (size * price + upsize * upprice) / (size + upsize)
    assert tr.commission == commission + upcomm
    size = tr.size
    price = tr.price
    commission = tr.commission
    upsize = -size
    upprice = 12.5
    upvalue = upsize * upprice
    upcomm = abs(value) * commrate
    tr.update(order=order, size=upsize, price=upprice, value=upvalue, commission=upcomm, pnl=0.0, comminfo=FakeCommInfo())
    assert tr.isclosed
    assert tr.size == size + upsize
    assert tr.price == price
    assert tr.commission == commission + upcomm
if __name__ == '__main__':
    test_run(main=True)
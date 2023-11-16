from __future__ import absolute_import, division, print_function, unicode_literals
import itertools
import time
try:
    time_clock = time.process_time
except:
    time_clock = time.clock
import testcommon
from backtrader.utils.py3 import range
import backtrader as bt
import backtrader.indicators as btind
CHKVALUES = ['14525.80', '14525.80', '15408.20', '15408.20', '14763.90', '14763.90', '14763.90', '14763.90', '14763.90', '14763.90', '14763.90', '14763.90', '14763.90', '14763.90', '13187.10', '13187.10', '13187.10', '13684.40', '13684.40', '13684.40', '13684.40', '13684.40', '13684.40', '13656.10', '13656.10', '13656.10', '13656.10', '12988.10', '12988.10', '12988.10', '12988.10', '12988.10', '12988.10', '12988.10', '12988.10', '12988.10', '12988.10', '12988.10', '12988.10', '12988.10']
CHKCASH = ['13525.80', '13525.80', '14408.20', '14408.20', '13763.90', '13763.90', '13763.90', '13763.90', '13763.90', '13763.90', '13763.90', '13763.90', '13763.90', '13763.90', '12187.10', '12187.10', '12187.10', '12684.40', '12684.40', '12684.40', '12684.40', '12684.40', '12684.40', '12656.10', '12656.10', '12656.10', '12656.10', '11988.10', '11988.10', '11988.10', '11988.10', '11988.10', '11988.10', '11988.10', '11988.10', '11988.10', '11988.10', '11988.10', '11988.10', '11988.10']
_chkvalues = []
_chkcash = []

class TestStrategy(bt.Strategy):
    params = (('period', 15), ('printdata', True), ('printops', True))

    def log(self, txt, dt=None):
        if False:
            i = 10
            return i + 15
        dt = dt or self.data.datetime[0]
        dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.orderid = None
        self.sma = btind.SMA(self.data, period=self.p.period)
        self.cross = btind.CrossOver(self.data.close, self.sma, plot=True)

    def start(self):
        if False:
            while True:
                i = 10
        self.broker.setcommission(commission=2.0, mult=10.0, margin=1000.0)
        self.tstart = time_clock()
        self.buy_create_idx = itertools.count()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        global _chkvalues
        global _chkcash
        tused = time_clock() - self.tstart
        if self.p.printdata:
            self.log('Time used: %s  - Period % d - Start value: %.2f - End value: %.2f' % (str(tused), self.p.period, self.broker.startingcash, self.broker.getvalue()))
        value = '%.2f' % self.broker.getvalue()
        _chkvalues.append(value)
        cash = '%.2f' % self.broker.getcash()
        _chkcash.append(cash)

    def next(self):
        if False:
            print('Hello World!')
        if self.orderid:
            return
        if not self.position.size:
            if self.cross > 0.0:
                self.orderid = self.buy()
        elif self.cross < 0.0:
            self.orderid = self.close()
chkdatas = 1

def test_run(main=False):
    if False:
        print('Hello World!')
    global _chkvalues
    global _chkcash
    for runonce in [True, False]:
        for preload in [True, False]:
            for exbar in [True, False, -1, -2]:
                _chkvalues = list()
                _chkcash = list()
                datas = [testcommon.getdata(i) for i in range(chkdatas)]
                testcommon.runtest(datas, TestStrategy, runonce=runonce, preload=preload, exbar=exbar, optimize=True, period=range(5, 45), printdata=main, printops=main, plot=False)
                if not main:
                    assert CHKVALUES == _chkvalues
                    assert CHKCASH == _chkcash
                else:
                    print('*' * 50)
                    print(CHKVALUES == _chkvalues)
                    print('-' * 50)
                    print(CHKVALUES)
                    print('-' * 50)
                    print(_chkvalues)
                    print('*' * 50)
                    print(CHKCASH == _chkcash)
                    print('-' * 50)
                    print(CHKCASH)
                    print('-' * 50)
                    print(_chkcash)
if __name__ == '__main__':
    test_run(main=True)
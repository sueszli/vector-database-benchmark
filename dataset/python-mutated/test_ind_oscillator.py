from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chkvals = [['56.477000', '51.185333', '2.386667']]
chkmin = 30
chkind = btind.Oscillator

class TS2(testcommon.TestStrategy):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        ind = btind.MovAv.SMA(self.data)
        self.p.inddata = [ind]
        super(TS2, self).__init__()

def test_run(main=False):
    if False:
        print('Hello World!')
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, TS2, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals)
if __name__ == '__main__':
    test_run(main=True)
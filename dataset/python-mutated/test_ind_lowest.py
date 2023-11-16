from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chkvals = [['4019.890000', '3570.170000', '3506.070000']]
chkmin = 14
chkind = btind.Lowest
chkargs = dict(period=14)

def test_run(main=False):
    if False:
        for i in range(10):
            print('nop')
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals, chkargs=chkargs)
if __name__ == '__main__':
    test_run(main=True)
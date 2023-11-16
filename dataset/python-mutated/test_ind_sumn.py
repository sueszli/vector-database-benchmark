from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chkvals = [['57406.490000', '50891.010000', '50424.690000']]
chkmin = 14
chkind = btind.SumN
chkargs = dict(period=14)

def test_run(main=False):
    if False:
        return 10
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals, chkargs=chkargs)
if __name__ == '__main__':
    test_run(main=True)
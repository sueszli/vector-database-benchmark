from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chkvals = [['3836.453333', '3703.962333', '3741.802000']]
chkmin = 30
chkind = [btind.SMA]
chkargs = dict()

def test_run(main=False):
    if False:
        print('Hello World!')
    for runonce in [True, False]:
        data = testcommon.getdata(0)
        data.resample(timeframe=bt.TimeFrame.Weeks, compression=1)
        datas = [data]
        testcommon.runtest(datas, testcommon.TestStrategy, main=main, runonce=runonce, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals, chkargs=chkargs)
if __name__ == '__main__':
    test_run(main=True)
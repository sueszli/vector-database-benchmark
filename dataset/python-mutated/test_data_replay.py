from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chknext = 113
chkvals = [['3836.453333', '3703.962333', '3741.802000']]
chkmin = 30
chkind = [btind.SMA]
chkargs = dict()

def test_run(main=False, exbar=False):
    if False:
        for i in range(10):
            print('nop')
    data = testcommon.getdata(0)
    data.replay(timeframe=bt.TimeFrame.Weeks, compression=1)
    datas = [data]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals, chknext=chknext, chkargs=chkargs, runonce=False, preload=False, exbar=exbar)
if __name__ == '__main__':
    for exbar in [False, -1, -2]:
        test_run(main=True, exbar=exbar)
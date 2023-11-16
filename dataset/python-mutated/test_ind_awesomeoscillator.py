from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
chkdatas = 1
chkvals = [['50.804206', '72.983735', '33.655941']]
chkmin = 34
chkind = bt.ind.AO

def test_run(main=False):
    if False:
        return 10
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals)
if __name__ == '__main__':
    test_run(main=True)
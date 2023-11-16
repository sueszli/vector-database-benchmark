from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
chkdatas = 1
chkvals = [['67.786097', '59.856230', '38.287526']]
chkmin = 25
chkind = bt.ind.RMI

def test_run(main=False):
    if False:
        i = 10
        return i + 15
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals)
if __name__ == '__main__':
    test_run(main=True)
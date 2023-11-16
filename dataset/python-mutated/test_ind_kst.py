from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chkvals = [['18.966300', '33.688645', '27.643797'], ['11.123593', '37.882890', '16.602624']]
chkmin = 48
chkind = bt.ind.KST

def test_run(main=False):
    if False:
        print('Hello World!')
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals)
if __name__ == '__main__':
    test_run(main=True)
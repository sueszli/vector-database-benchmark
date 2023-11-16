from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chkvals = [['4115.563246', '3852.837209', '3665.728415'], ['4218.452327', '3949.158140', '3757.371626'], ['4012.674165', '3756.516279', '3574.085205']]
chkmin = 59
chkind = btind.DEMAEnvelope

def test_run(main=False):
    if False:
        while True:
            i = 10
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals)
if __name__ == '__main__':
    test_run(main=True)
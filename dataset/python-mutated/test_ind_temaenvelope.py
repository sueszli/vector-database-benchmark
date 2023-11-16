from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chkvals = [['4113.721705', '3862.386854', '3832.691054'], ['4216.564748', '3958.946525', '3928.508331'], ['4010.878663', '3765.827182', '3736.873778']]
chkmin = 88
chkind = btind.TEMAEnvelope

def test_run(main=False):
    if False:
        print('Hello World!')
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals)
if __name__ == '__main__':
    test_run(main=True)
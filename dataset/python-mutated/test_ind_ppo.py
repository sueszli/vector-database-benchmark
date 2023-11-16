from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chkvals = [['0.633439', '0.883552', '0.049430'], ['0.540516', '0.724136', '-0.079820'], ['0.092923', '0.159416', '0.129250']]
chkmin = 34
chkind = btind.PPO

def test_run(main=False):
    if False:
        while True:
            i = 10
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals)
if __name__ == '__main__':
    test_run(main=True)
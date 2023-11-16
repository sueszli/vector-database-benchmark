from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
import backtrader.indicators as btind
chkdatas = 1
chkvals = [['0.629452', '0.875813', '0.049405'], ['0.537193', '0.718852', '-0.080645'], ['0.092259', '0.156962', '0.130050']]
chkmin = 34
chkind = btind.PPOShort

def test_run(main=False):
    if False:
        while True:
            i = 10
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals)
if __name__ == '__main__':
    test_run(main=True)
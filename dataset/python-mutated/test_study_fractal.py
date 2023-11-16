from __future__ import absolute_import, division, print_function, unicode_literals
import testcommon
import backtrader as bt
chkdatas = 1
chkvals = [['nan', 'nan', 'nan'], ['nan', 'nan', '3553.692850']]
chkmin = 5
chkind = bt.studies.Fractal

def test_run(main=False):
    if False:
        while True:
            i = 10
    datas = [testcommon.getdata(i) for i in range(chkdatas)]
    testcommon.runtest(datas, testcommon.TestStrategy, main=main, plot=main, chkind=chkind, chkmin=chkmin, chkvals=chkvals)
if __name__ == '__main__':
    test_run(main=True)
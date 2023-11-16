from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt
import datetime

class St(bt.Strategy):

    def logdata(self):
        if False:
            for i in range(10):
                print('nop')
        txt = []
        txt.append('{}'.format(len(self)))
        txt.append('{}'.format(self.data.datetime.datetime(0).isoformat()))
        txt.append(' open BID: ' + '{}'.format(self.datas[0].open[0]))
        txt.append(' open ASK: ' + '{}'.format(self.datas[1].open[0]))
        txt.append(' high BID: ' + '{}'.format(self.datas[0].high[0]))
        txt.append(' high ASK: ' + '{}'.format(self.datas[1].high[0]))
        txt.append(' low BID: ' + '{}'.format(self.datas[0].low[0]))
        txt.append(' low ASK: ' + '{}'.format(self.datas[1].low[0]))
        txt.append(' close BID: ' + '{}'.format(self.datas[0].close[0]))
        txt.append(' close ASK: ' + '{}'.format(self.datas[1].close[0]))
        txt.append(' volume: ' + '{:.2f}'.format(self.data.volume[0]))
        print(','.join(txt))
    data_live = False

    def notify_data(self, data, status, *args, **kwargs):
        if False:
            print('Hello World!')
        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status), *args)
        if self.datas[0]._laststatus == self.datas[0].LIVE and self.datas[1]._laststatus == self.datas[1].LIVE:
            self.data_live = True

    def next(self):
        if False:
            i = 10
            return i + 15
        self.logdata()
        if not self.data_live:
            return
ib_symbol = 'EUR.USD-CASH-IDEALPRO'
compression = 5

def run(args=None):
    if False:
        return 10
    cerebro = bt.Cerebro(stdstats=False)
    store = bt.stores.IBStore(port=7497)
    data0 = store.getdata(dataname=ib_symbol, timeframe=bt.TimeFrame.Ticks)
    cerebro.resampledata(data0, timeframe=bt.TimeFrame.Seconds, compression=compression)
    data1 = store.getdata(dataname=ib_symbol, timeframe=bt.TimeFrame.Ticks, what='ASK')
    cerebro.resampledata(data1, timeframe=bt.TimeFrame.Seconds, compression=compression)
    cerebro.broker = store.getbroker()
    cerebro.addstrategy(St)
    cerebro.run()
if __name__ == '__main__':
    run()
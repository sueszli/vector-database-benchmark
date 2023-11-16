__author__ = 'mayanqiong'
'\n时间帮助函数，根据回测/实盘获取不同的当前时间\n\n模块内部创建使用\n'
import time

class TqDatetimeState:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.data_ready = False
        self.tqsdk_backtest = {}

    def get_current_dt(self):
        if False:
            print('Hello World!')
        '返回当前 nano timestamp'
        if self.tqsdk_backtest:
            return self.tqsdk_backtest.get('current_dt')
        else:
            return int(time.time() * 1000000) * 1000

    def update_state(self, diff):
        if False:
            for i in range(10):
                print('nop')
        self.tqsdk_backtest.update(diff.get('_tqsdk_backtest', {}))
        if not self.data_ready and diff.get('mdhis_more_data', True) is False:
            self.data_ready = True
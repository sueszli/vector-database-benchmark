__author__ = 'limin'
import pandas as pd
import datetime
from contextlib import closing
from tqsdk import TqApi, TqAuth, TqBacktest, BacktestFinished, TargetPosTask
from tqsdk.tafunc import sma, ema2, trma
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
'\n应用随机森林对某交易日涨跌情况的预测(使用sklearn包)\n参考:https://www.joinquant.com/post/1571\n注: 该示例策略仅用于功能示范, 实盘时请根据自己的策略/经验进行修改\n'
symbol = 'SHFE.ru1811'
(close_hour, close_minute) = (14, 50)

def get_prediction_data(klines, n):
    if False:
        for i in range(10):
            print('nop')
    '获取用于随机森林的n个输入数据(n为数据长度): n天中每天的特征参数及其涨跌情况'
    close_prices = klines.close[-30 - n:]
    sma_data = sma(close_prices, 30, 0.02)[-n:]
    wma_data = ema2(close_prices, 30)[-n:]
    mom_data = trma(close_prices, 30)[-n:]
    x_all = list(zip(sma_data, wma_data, mom_data))
    y_all = list((klines.close.iloc[i] >= klines.close.iloc[i - 1] for i in list(reversed(range(-1, -n - 1, -1)))))
    x_train = x_all[:-1]
    x_predict = x_all[-1]
    y_train = y_all[1:]
    return (x_train, y_train, x_predict)
predictions = []
api = TqApi(backtest=TqBacktest(start_dt=datetime.date(2018, 7, 2), end_dt=datetime.date(2018, 9, 26)), auth=TqAuth('快期账户', '账户密码'))
quote = api.get_quote(symbol)
klines = api.get_kline_serial(symbol, duration_seconds=24 * 60 * 60)
target_pos = TargetPosTask(api, symbol)
with closing(api):
    try:
        while True:
            while not api.is_changing(klines.iloc[-1], 'datetime'):
                api.wait_update()
            while True:
                api.wait_update()
                if api.is_changing(quote, 'datetime'):
                    now = datetime.datetime.strptime(quote.datetime, '%Y-%m-%d %H:%M:%S.%f')
                    if now.hour == close_hour and now.minute >= close_minute:
                        (x_train, y_train, x_predict) = get_prediction_data(klines, 75)
                        clf = RandomForestClassifier(n_estimators=30, bootstrap=True)
                        clf.fit(x_train, y_train)
                        predictions.append(bool(clf.predict([x_predict])))
                        if predictions[-1] == True:
                            print(quote.datetime, '预测下一交易日为 涨')
                            target_pos.set_target_volume(10)
                        else:
                            print(quote.datetime, '预测下一交易日为 跌')
                            target_pos.set_target_volume(-10)
                        break
    except BacktestFinished:
        klines['pre_close'] = klines['close'].shift(1)
        klines = klines[-len(predictions) + 1:]
        klines['prediction'] = predictions[:-1]
        results = (klines['close'] - klines['pre_close'] >= 0) == klines['prediction']
        print(klines)
        print('----回测结束----')
        print('预测结果正误:\n', results)
        print('预测结果数目统计: 总计', len(results), '个预测结果')
        print(pd.value_counts(results))
        print('预测的准确率:')
        print(pd.value_counts(results)[True] / len(results))
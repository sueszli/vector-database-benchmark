__author__ = 'limin'
'\nVolume Weighted Average Price策略 (难度：高级)\n参考: https://www.shinnytech.com/blog/vwap\n注: 该示例策略仅用于功能示范, 实盘时请根据自己的策略/经验进行修改\n'
import datetime
from tqsdk import TqApi, TqAuth, TargetPosTask
TIME_CELL = 5 * 60
TARGET_VOLUME = 300
SYMBOL = 'DCE.jd2001'
HISTORY_DAY_LENGTH = 20
(START_HOUR, START_MINUTE) = (9, 35)
(END_HOUR, END_MINUTE) = (10, 50)
api = TqApi(auth=TqAuth('快期账户', '账户密码'))
print('策略开始运行')
time_slot_start = datetime.time(START_HOUR, START_MINUTE)
time_slot_end = datetime.time(END_HOUR, END_MINUTE)
klines = api.get_kline_serial(SYMBOL, TIME_CELL, data_length=int(10 * 60 * 60 / TIME_CELL * HISTORY_DAY_LENGTH))
target_pos = TargetPosTask(api, SYMBOL)
position = api.get_position(SYMBOL)

def get_kline_time(kline_datetime):
    if False:
        while True:
            i = 10
    '获取k线的时间(不包含日期)'
    kline_time = datetime.datetime.fromtimestamp(kline_datetime // 1000000000).time()
    return kline_time

def get_market_day(kline_datetime):
    if False:
        while True:
            i = 10
    '获取k线所对应的交易日'
    kline_dt = datetime.datetime.fromtimestamp(kline_datetime // 1000000000)
    if kline_dt.hour >= 18:
        kline_dt = kline_dt + datetime.timedelta(days=1)
    while kline_dt.weekday() >= 5:
        kline_dt = kline_dt + datetime.timedelta(days=1)
    return kline_dt.date()
klines['time'] = klines.datetime.apply(lambda x: get_kline_time(x))
klines['date'] = klines.datetime.apply(lambda x: get_market_day(x))
if time_slot_end > time_slot_start:
    klines = klines[(klines['time'] >= time_slot_start) & (klines['time'] <= time_slot_end)]
else:
    klines = klines[(klines['time'] >= time_slot_start) | (klines['time'] <= time_slot_end)]
date_cnt = klines['date'].value_counts()
max_num = date_cnt.max()
need_date = date_cnt[date_cnt == max_num].sort_index().index[-HISTORY_DAY_LENGTH - 1:-1]
df = klines[klines['date'].isin(need_date)]
datetime_grouped = df.groupby(['date', 'time'])['volume'].sum()
volume_percent = datetime_grouped / datetime_grouped.groupby(level=0).sum()
predicted_percent = volume_percent.groupby(level=1).mean()
print('各时间单元成交量占比: %s' % predicted_percent)
predicted_volume = {}
percentage_left = 1
volume_left = TARGET_VOLUME
for (index, value) in predicted_percent.items():
    volume = round(volume_left * (value / percentage_left))
    predicted_volume[index] = volume
    percentage_left -= value
    volume_left -= volume
print('各时间单元应下单手数: %s' % predicted_volume)
current_volume = 0
while True:
    api.wait_update()
    if api.is_changing(klines.iloc[-1], 'datetime'):
        t = datetime.datetime.fromtimestamp(klines.iloc[-1]['datetime'] // 1000000000).time()
        if t in predicted_volume:
            current_volume += predicted_volume[t]
            print('到达下一时间单元,调整持仓为: %d' % current_volume)
            target_pos.set_target_volume(current_volume)
    if api.is_changing(position, 'volume_long') or api.is_changing(position, 'volume_short'):
        if position['volume_long'] - position['volume_short'] == TARGET_VOLUME:
            break
api.close()
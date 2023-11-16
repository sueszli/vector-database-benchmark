__author__ = 'mayanqiong'
import os
import shutil
import struct
import numpy as np
import pandas as pd
from filelock import FileLock
from tqsdk.channel import TqChan
from tqsdk.diff import _get_obj
from tqsdk.rangeset import _rangeset_difference, _rangeset_intersection
from tqsdk.tafunc import get_dividend_df
from tqsdk.utils import _generate_uuid
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.tqsdk/data_series_1')

class DataSeries:
    """
    获取数据模块，支持用户获取任意时间段内的 Kline / Tick 数据，返回格式为 pandas.DataFrame

    支持用户设置开始时间和结束时间，且返回的 df 不会随着行情更新

    首个版本功能点：
    1. 缓存数据，目录 ~/.tqsdk/data_series，文件名规则为 symbol.dur.start_id.end_id                                                                          
    2. 每次下载数据先检查缓存数据是否包含需要下载数据，计算出还需要下载的数据段
    3. 如果有需要下载的数据段
       下载的数据先写在 symbol.dur.temp 文件，下载完成之后文件名修改为 symbol.dur.start_id.end_id
    4. 下载完数据之后，合并连续的数据文件，更新缓存文件
    5. 如果有复权参数，计算复权数据
    6. 返回完整的 pandas.DataFrame 给用户


    **功能限制说明**
    * 该接口返回的 df 不会随着行情更新
    * 暂不支持多合约 Kline 缓存
    * 不支持用户回测/复盘使用。get_data_series() 是直接连接行情服务器，是会下载到未来数据的。
    * 不支持多进程/线程/协程。每个合约+周期只能在同一个线程/进程里下载，因为需要写/读/修改文件，多个线程/进程会造成冲突。

    """

    def __init__(self, api, symbol_list, dur_nano, start_dt_nano, end_dt_nano, adj_type=None) -> None:
        if False:
            print('Hello World!')
        '\n        创建历史数据下载器实例\n\n        Args:\n            api (TqApi): TqApi实例，该下载器将使用指定的api下载数据\n\n            symbol_list: 需要下载数据的合约代码，当指定多个合约代码时将其他合约按第一个合约的交易时间对齐\n\n            dur_nano (int): 数据周期，纳秒数\n\n            start_dt_nano (int): 起始时间, 纳秒数\n\n            end_dt_nano (int): 结束时间, 纳秒数\n\n            adj_type (str/None): 复权计算方式，默认值为 None。"F" 为前复权；"B" 为后复权；None 表示不复权。只对股票、基金合约有效。\n        '
        self._api = api
        self._symbol_list = symbol_list if isinstance(symbol_list, list) else [symbol_list]
        self._dur_nano = dur_nano
        self._start_dt_nano = start_dt_nano
        self._end_dt_nano = end_dt_nano + 1
        self._adj_type = adj_type
        self._dividend_cache = {}
        self.df = pd.DataFrame()
        self.is_ready = False
        DataSeries._ensure_cache_dir()
        self._api.create_task(self._run())

    async def _run(self):
        symbol = self._symbol_list[0]
        lock_path = DataSeries._get_lock_path(symbol, self._dur_nano)
        with FileLock(lock_path, timeout=-1):
            rangeset_id = DataSeries._get_rangeset_id(symbol, self._dur_nano)
            rangeset_dt = DataSeries._get_rangeset_dt(symbol, self._dur_nano, rangeset_id)
            DataSeries._assert_rangeset_asce_sorted(rangeset_id)
            DataSeries._assert_rangeset_asce_sorted(rangeset_dt)
            if len(rangeset_dt) > 0:
                rangeset_dt[-1] = (rangeset_dt[-1][0], rangeset_dt[-1][1] - self._dur_nano)
                if rangeset_dt[-1][0] == rangeset_dt[-1][1]:
                    rangeset_dt.pop(-1)
            diff_rangeset = _rangeset_difference([(self._start_dt_nano, self._end_dt_nano)], rangeset_dt)
            if len(diff_rangeset) > 0:
                await self._download_data_series(diff_rangeset)
                self._merge_rangeset()
                rangeset_id = DataSeries._get_rangeset_id(symbol, self._dur_nano)
                rangeset_dt = DataSeries._get_rangeset_dt(symbol, self._dur_nano, rangeset_id)
                DataSeries._assert_rangeset_asce_sorted(rangeset_id)
                DataSeries._assert_rangeset_asce_sorted(rangeset_dt)
            assert len(rangeset_id) == len(rangeset_dt) > 0
            target_rangeset_dt = _rangeset_intersection([(self._start_dt_nano, self._end_dt_nano)], rangeset_dt)
            assert len(target_rangeset_dt) <= 1
            if len(target_rangeset_dt) == 0:
                self.is_ready = True
                return
            (start_dt, end_dt) = target_rangeset_dt[0]
            range_id = None
            for index in range(len(rangeset_dt)):
                range_dt = rangeset_dt[index]
                if range_dt[0] <= start_dt and end_dt <= range_dt[1]:
                    range_id = rangeset_id[index]
                    break
            assert range_id
            filename = os.path.join(CACHE_DIR, f'{symbol}.{self._dur_nano}.{range_id[0]}.{range_id[1]}')
            data_cols = DataSeries._get_data_cols(symbol=self._symbol_list[0], dur_nano=self._dur_nano)
            dtype = np.dtype([('id', 'i8'), ('datetime', 'i8')] + [(col, 'f8') for col in data_cols])
            fp = np.memmap(filename, dtype=dtype, mode='r', shape=range_id[1] - range_id[0])
            start_id = fp[fp['datetime'] <= start_dt][-1]['id']
            end_id = fp[fp['datetime'] < end_dt][-1]['id']
            rows = end_id - start_id + 1
            array = fp[start_id - range_id[0]:start_id - range_id[0] + rows]
            self.df['id'] = array['id']
            self.df['datetime'] = array['datetime']
            for c in data_cols:
                self.df[c] = array[c]
            self.df['symbol'] = symbol
            self.df['duration'] = self._dur_nano
            quote = self._api.get_quote(symbol)
            if self._adj_type and quote.ins_class in ['STOCK', 'FUND']:
                factor_df = await self._update_dividend_factor(symbol)
                if self._adj_type == 'F':
                    for i in range(factor_df.shape[0] - 1, -1, -1):
                        dt = factor_df.iloc[i].datetime
                        factor = factor_df.iloc[i].factor
                        adj_cols = DataSeries._get_adj_cols(symbol, self._dur_nano)
                        lt = self.df['datetime'].lt(dt)
                        for col in adj_cols:
                            self.df.loc[lt, col] = self.df.loc[lt, col] * factor
                if self._adj_type == 'B':
                    for i in range(factor_df.shape[0]):
                        dt = factor_df.iloc[i].datetime
                        factor = factor_df.iloc[i].factor
                        adj_cols = DataSeries._get_adj_cols(symbol, self._dur_nano)
                        ge = self.df['datetime'].ge(dt)
                        self.df.loc[ge, adj_cols] = self.df.loc[ge, adj_cols] / factor
            self.is_ready = True

    async def _download_data_series(self, rangeset):
        symbol = self._symbol_list[0]
        for (start_dt, end_dt) in rangeset:
            try:
                (start_id, end_id) = (None, None)
                temp_filename = os.path.join(CACHE_DIR, f'{symbol}.{self._dur_nano}.temp')
                temp_file = open(temp_filename, 'wb')
                data_chan = TqChan(self._api)
                task = self._api.create_task(self._download_data(start_dt, end_dt, data_chan))
                async for item in data_chan:
                    temp_file.write(struct.pack('@qq' + 'd' * (len(item) - 2), *item))
                    if start_id is None:
                        start_id = item[0]
                    end_id = item[0]
            except Exception as e:
                temp_file.close()
            else:
                temp_file.close()
                if start_id is not None and end_id is not None:
                    target_filename = os.path.join(CACHE_DIR, f'{symbol}.{self._dur_nano}.{start_id}.{end_id + 1}')
                    shutil.move(temp_filename, target_filename)
            finally:
                task.cancel()
                await task

    async def _download_data(self, start_dt, end_dt, data_chan):
        symbol = self._symbol_list[0]
        chart_info = {'aid': 'set_chart', 'chart_id': _generate_uuid('PYSDK_data_series'), 'ins_list': symbol, 'duration': int(self._dur_nano), 'view_width': 2000, 'focus_datetime': int(start_dt), 'focus_position': 0}
        self._api._send_chan.send_nowait(chart_info)
        chart = _get_obj(self._api._data, ['charts', chart_info['chart_id']])
        current_id = None
        path = ['klines', symbol, str(self._dur_nano)] if self._dur_nano != 0 else ['ticks', symbol]
        serial = _get_obj(self._api._data, path)
        cols = DataSeries._get_data_cols(symbol, self._dur_nano)
        try:
            async with self._api.register_update_notify() as update_chan:
                async for _ in update_chan:
                    if not chart_info.items() <= _get_obj(chart, ['state']).items():
                        continue
                    left_id = chart.get('left_id', -1)
                    right_id = chart.get('right_id', -1)
                    if left_id == -1 and right_id == -1 or self._api._data.get('mdhis_more_data', True):
                        continue
                    if serial.get('last_id', -1) == -1:
                        continue
                    if current_id is None:
                        current_id = max(left_id, 0)
                    while current_id <= right_id:
                        item = serial['data'].get(str(current_id), {})
                        if item.get('datetime', 0) == 0 or item['datetime'] >= end_dt:
                            return
                        row = [current_id, item['datetime']] + [DataSeries._get_float_value(item, c) for c in cols]
                        await data_chan.send(row)
                        current_id += 1
                        self._current_dt_nano = item['datetime']
                    chart_info.pop('focus_datetime', None)
                    chart_info.pop('focus_position', None)
                    chart_info['left_kline_id'] = current_id
                    await self._api._send_chan.send(chart_info)
        finally:
            await data_chan.close()
            await self._api._send_chan.send({'aid': 'set_chart', 'chart_id': chart_info['chart_id'], 'ins_list': '', 'duration': self._dur_nano, 'view_width': 2000})

    async def _update_dividend_factor(self, symbol):
        quote = self._api.get_quote(symbol)
        df = get_dividend_df(quote.stock_dividend_ratio, quote.cash_dividend_ratio)
        between = df['datetime'].between(self._start_dt_nano, self._end_dt_nano)
        df['pre_close'] = float('nan')
        for i in df[between].index:
            chart_info = {'aid': 'set_chart', 'chart_id': _generate_uuid('PYSDK_data_factor'), 'ins_list': symbol, 'duration': 86400 * 1000000000, 'view_width': 2, 'focus_datetime': int(df.iloc[i].datetime), 'focus_position': 1}
            await self._api._send_chan.send(chart_info)
            chart = _get_obj(self._api._data, ['charts', chart_info['chart_id']])
            serial = _get_obj(self._api._data, ['klines', symbol, str(86400000000000)])
            try:
                async with self._api.register_update_notify() as update_chan:
                    async for _ in update_chan:
                        if not chart_info.items() <= _get_obj(chart, ['state']).items():
                            continue
                        left_id = chart.get('left_id', -1)
                        right_id = chart.get('right_id', -1)
                        if left_id == -1 and right_id == -1 or self._api._data.get('mdhis_more_data', True) or serial.get('last_id', -1) == -1:
                            continue
                        last_item = serial['data'].get(str(left_id), {})
                        df.loc[i, 'pre_close'] = last_item['close'] if last_item.get('close') else float('nan')
                        break
            finally:
                await self._api._send_chan.send({'aid': 'set_chart', 'chart_id': chart_info['chart_id'], 'ins_list': '', 'duration': 86400000000000, 'view_width': 2})
        df['factor'] = (df['pre_close'] - df['cash_dividend']) / df['pre_close'] / (1 + df['stock_dividend'])
        df['factor'].fillna(1, inplace=True)
        return df

    def _merge_rangeset(self):
        if False:
            for i in range(10):
                print('nop')
        symbol = self._symbol_list[0]
        rangeset = DataSeries._get_rangeset_id(symbol, self._dur_nano)
        if len(rangeset) <= 1:
            return
        rangset_group = [[rangeset[0] + (rangeset[0][1] - rangeset[0][0],)]]
        for i in range(1, len(rangeset)):
            last_r = rangeset[i - 1]
            r = rangeset[i]
            assert (r[0] < r[1]) & (last_r[0] < last_r[1])
            if i == len(rangeset) - 1:
                assert last_r[1] - 1 <= r[0]
                if last_r[1] == r[0]:
                    rangset_group[-1].append(r + (r[1] - r[0],))
                elif last_r[1] - 1 == r[0]:
                    rangset_group[-1][-1] = last_r + (last_r[1] - 1 - last_r[0],)
                    rangset_group[-1].append(r + (r[1] - r[0],))
                else:
                    rangset_group.append([r + (r[1] - r[0],)])
            else:
                assert last_r[1] <= r[0]
                if last_r[1] == r[0]:
                    rangset_group[-1].append(r + (r[1] - r[0],))
                else:
                    rangset_group.append([r + (r[1] - r[0],)])
        data_cols = DataSeries._get_data_cols(symbol, self._dur_nano)
        dtype = np.dtype([('id', 'i8'), ('datetime', 'i8')] + [(col, 'f8') for col in data_cols])
        for rangeset in rangset_group:
            if len(rangeset) == 1:
                continue
            (first_r_0, first_r_1, first_r_rows) = rangeset[0]
            temp_filename = os.path.join(CACHE_DIR, f'{symbol}.{self._dur_nano}.{first_r_0}.{first_r_1}')
            all_rows = first_r_rows
            last_r_1 = None
            for (s, e, rows_number) in rangeset[1:]:
                filename = os.path.join(CACHE_DIR, f'{symbol}.{self._dur_nano}.{s}.{e}')
                fp = np.memmap(filename, dtype=dtype, mode='r+', shape=rows_number)
                temp_fp = np.memmap(temp_filename, dtype=dtype, mode='r+', offset=dtype.itemsize * all_rows, shape=rows_number)
                temp_fp[0:rows_number] = fp[0:rows_number]
                temp_fp._mmap.close()
                fp._mmap.close()
                os.remove(filename)
                all_rows += rows_number
                last_r_1 = e
            os.rename(temp_filename, os.path.join(CACHE_DIR, f'{symbol}.{self._dur_nano}.{first_r_0}.{last_r_1}'))

    @staticmethod
    def _assert_rangeset_asce_sorted(rangeset):
        if False:
            while True:
                i = 10
        assert all([(rangeset[i][0] < rangeset[i][1]) & (True if i == 0 else rangeset[i - 1][1] < rangeset[i][0]) for i in range(len(rangeset))])

    @staticmethod
    def _get_rangeset_id(symbol, dur_nano):
        if False:
            while True:
                i = 10
        rangeset_id = []
        for filename in os.listdir(CACHE_DIR):
            key = f'{symbol}.{dur_nano}.'
            if os.path.isfile(os.path.join(CACHE_DIR, filename)) and filename.startswith(key) and ('temp' not in filename):
                (start_id, end_id) = [int(i) for i in filename.split(key)[1].split('.')[-2:]]
                rangeset_id.append((start_id, end_id))
        rangeset_id.sort()
        return rangeset_id

    @staticmethod
    def _get_rangeset_dt(symbol, dur_nano, rangeset_id):
        if False:
            return 10
        rangeset_dt = []
        cols = DataSeries._get_data_cols(symbol, dur_nano)
        dtype = np.dtype([('id', 'i8'), ('datetime', 'i8')] + [(col, 'f8') for col in cols])
        for (start_id, end_id) in rangeset_id:
            filename = os.path.join(CACHE_DIR, f'{symbol}.{dur_nano}.{start_id}.{end_id}')
            fp = np.memmap(filename, dtype=dtype, mode='r', shape=end_id - start_id)
            (first_dt, last_dt) = (fp[0]['datetime'], fp[-1]['datetime'])
            rangeset_dt.append((first_dt, last_dt + (dur_nano if dur_nano > 0 else 100)))
        return rangeset_dt

    @staticmethod
    def _ensure_cache_dir():
        if False:
            print('Hello World!')
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)

    @staticmethod
    def _get_lock_path(symbol, dur_nano):
        if False:
            return 10
        return os.path.join(CACHE_DIR, f'.{symbol}.{dur_nano}.lock')

    @staticmethod
    def _get_data_cols(symbol, dur_nano):
        if False:
            return 10
        if dur_nano != 0:
            return ['open', 'high', 'low', 'close', 'volume', 'open_oi', 'close_oi']
        else:
            cols = ['last_price', 'highest', 'lowest', 'average', 'volume', 'amount', 'open_interest']
            price_length = 5 if symbol.split('.')[0] in {'SHFE', 'SSE', 'SZSE'} else 1
            for i in range(1, price_length + 1):
                cols.extend((f'{x}{i}' for x in ['bid_price', 'bid_volume', 'ask_price', 'ask_volume']))
            return cols

    @staticmethod
    def _get_adj_cols(symbol, dur_nano):
        if False:
            while True:
                i = 10
        if dur_nano != 0:
            return ['open', 'high', 'low', 'close']
        else:
            cols = ['last_price', 'highest', 'lowest', 'average']
            price_length = 5 if symbol.split('.')[0] in {'SHFE', 'SSE', 'SZSE'} else 1
            cols.extend((f'{x}{i}' for x in ['bid_price', 'ask_price'] for i in range(1, price_length + 1)))
            return cols

    @staticmethod
    def _get_float_value(obj, key) -> float:
        if False:
            i = 10
            return i + 15
        if key not in obj or isinstance(obj[key], str):
            return float('nan')
        return float(obj[key])
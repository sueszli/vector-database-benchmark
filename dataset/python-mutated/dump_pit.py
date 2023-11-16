"""
TODO:
- A more well-designed PIT database is required.
    - seperated insert, delete, update, query operations are required.
"""
import abc
import shutil
import struct
import traceback
from pathlib import Path
from typing import Iterable, List, Union
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from qlib.utils import fname_to_code, code_to_fname, get_period_offset
from qlib.config import C

class DumpPitData:
    PIT_DIR_NAME = 'financial'
    PIT_CSV_SEP = ','
    DATA_FILE_SUFFIX = '.data'
    INDEX_FILE_SUFFIX = '.index'
    INTERVAL_quarterly = 'quarterly'
    INTERVAL_annual = 'annual'
    PERIOD_DTYPE = C.pit_record_type['period']
    INDEX_DTYPE = C.pit_record_type['index']
    DATA_DTYPE = ''.join([C.pit_record_type['date'], C.pit_record_type['period'], C.pit_record_type['value'], C.pit_record_type['index']])
    NA_INDEX = C.pit_record_nan['index']
    INDEX_DTYPE_SIZE = struct.calcsize(INDEX_DTYPE)
    PERIOD_DTYPE_SIZE = struct.calcsize(PERIOD_DTYPE)
    DATA_DTYPE_SIZE = struct.calcsize(DATA_DTYPE)
    UPDATE_MODE = 'update'
    ALL_MODE = 'all'

    def __init__(self, csv_path: str, qlib_dir: str, backup_dir: str=None, freq: str='quarterly', max_workers: int=16, date_column_name: str='date', period_column_name: str='period', value_column_name: str='value', field_column_name: str='field', file_suffix: str='.csv', exclude_fields: str='', include_fields: str='', limit_nums: int=None):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Parameters\n        ----------\n        csv_path: str\n            stock data path or directory\n        qlib_dir: str\n            qlib(dump) data director\n        backup_dir: str, default None\n            if backup_dir is not None, backup qlib_dir to backup_dir\n        freq: str, default "quarterly"\n            data frequency\n        max_workers: int, default None\n            number of threads\n        date_column_name: str, default "date"\n            the name of the date field in the csv\n        file_suffix: str, default ".csv"\n            file suffix\n        include_fields: tuple\n            dump fields\n        exclude_fields: tuple\n            fields not dumped\n        limit_nums: int\n            Use when debugging, default None\n        '
        csv_path = Path(csv_path).expanduser()
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(',')
        if isinstance(include_fields, str):
            include_fields = include_fields.split(',')
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.file_suffix = file_suffix
        self.csv_files = sorted(csv_path.glob(f'*{self.file_suffix}') if csv_path.is_dir() else [csv_path])
        if limit_nums is not None:
            self.csv_files = self.csv_files[:int(limit_nums)]
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())
        self.works = max_workers
        self.date_column_name = date_column_name
        self.period_column_name = period_column_name
        self.value_column_name = value_column_name
        self.field_column_name = field_column_name
        self._mode = self.ALL_MODE

    def _backup_qlib_dir(self, target_dir: Path):
        if False:
            i = 10
            return i + 15
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def get_source_data(self, file_path: Path) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        df = pd.read_csv(str(file_path.resolve()), low_memory=False)
        df[self.value_column_name] = df[self.value_column_name].astype('float32')
        df[self.date_column_name] = df[self.date_column_name].str.replace('-', '').astype('int32')
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        if False:
            while True:
                i = 10
        return fname_to_code(file_path.name[:-len(self.file_suffix)].strip().lower())

    def get_dump_fields(self, df: Iterable[str]) -> Iterable[str]:
        if False:
            print('Hello World!')
        return set(self._include_fields) if self._include_fields else set(df[self.field_column_name]) - set(self._exclude_fields) if self._exclude_fields else set(df[self.field_column_name])

    def get_filenames(self, symbol, field, interval):
        if False:
            for i in range(10):
                print('nop')
        dir_name = self.qlib_dir.joinpath(self.PIT_DIR_NAME, symbol)
        dir_name.mkdir(parents=True, exist_ok=True)
        return (dir_name.joinpath(f'{field}_{interval[0]}{self.DATA_FILE_SUFFIX}'.lower()), dir_name.joinpath(f'{field}_{interval[0]}{self.INDEX_FILE_SUFFIX}'.lower()))

    def _dump_pit(self, file_path: str, interval: str='quarterly', overwrite: bool=False):
        if False:
            print('Hello World!')
        '\n        dump data as the following format:\n            `/path/to/<field>.data`\n                [date, period, value, _next]\n                [date, period, value, _next]\n                [...]\n            `/path/to/<field>.index`\n                [first_year, index, index, ...]\n\n        `<field.data>` contains the data as the point-in-time (PIT) order: `value` of `period`\n        is published at `date`, and its successive revised value can be found at `_next` (linked list).\n\n        `<field>.index` contains the index of value for each period (quarter or year). To save\n        disk space, we only store the `first_year` as its followings periods can be easily infered.\n\n        Parameters\n        ----------\n        symbol: str\n            stock symbol\n        interval: str\n            data interval\n        overwrite: bool\n            whether overwrite existing data or update only\n        '
        symbol = self.get_symbol_from_file(file_path)
        df = self.get_source_data(file_path)
        if df.empty:
            logger.warning(f'{symbol} file is empty')
            return
        for field in self.get_dump_fields(df):
            df_sub = df.query(f'{self.field_column_name}=="{field}"').sort_values(self.date_column_name)
            if df_sub.empty:
                logger.warning(f'field {field} of {symbol} is empty')
                continue
            (data_file, index_file) = self.get_filenames(symbol, field, interval)
            start_year = df_sub[self.period_column_name].min()
            end_year = df_sub[self.period_column_name].max()
            if interval == self.INTERVAL_quarterly:
                start_year //= 100
                end_year //= 100
            if not overwrite and index_file.exists():
                with open(index_file, 'rb') as fi:
                    (first_year,) = struct.unpack(self.PERIOD_DTYPE, fi.read(self.PERIOD_DTYPE_SIZE))
                    n_years = len(fi.read()) // self.INDEX_DTYPE_SIZE
                    if interval == self.INTERVAL_quarterly:
                        n_years //= 4
                    start_year = first_year + n_years
            else:
                with open(index_file, 'wb') as f:
                    f.write(struct.pack(self.PERIOD_DTYPE, start_year))
                first_year = start_year
            if start_year > end_year:
                logger.warning(f'{symbol}-{field} data already exists, continue to the next field')
                continue
            with open(index_file, 'ab') as fi:
                for year in range(start_year, end_year + 1):
                    if interval == self.INTERVAL_quarterly:
                        fi.write(struct.pack(self.INDEX_DTYPE * 4, *[self.NA_INDEX] * 4))
                    else:
                        fi.write(struct.pack(self.INDEX_DTYPE, self.NA_INDEX))
            if not overwrite and data_file.exists():
                with open(data_file, 'rb') as fd:
                    fd.seek(-self.DATA_DTYPE_SIZE, 2)
                    (last_date, _, _, _) = struct.unpack(self.DATA_DTYPE, fd.read())
                df_sub = df_sub.query(f'{self.date_column_name}>{last_date}')
            else:
                with open(data_file, 'wb+' if overwrite else 'ab+'):
                    pass
            with open(data_file, 'rb+') as fd, open(index_file, 'rb+') as fi:
                for (i, row) in df_sub.iterrows():
                    offset = get_period_offset(first_year, row.period, interval == self.INTERVAL_quarterly)
                    fi.seek(self.PERIOD_DTYPE_SIZE + self.INDEX_DTYPE_SIZE * offset)
                    (cur_index,) = struct.unpack(self.INDEX_DTYPE, fi.read(self.INDEX_DTYPE_SIZE))
                    if cur_index == self.NA_INDEX:
                        fi.seek(self.PERIOD_DTYPE_SIZE + self.INDEX_DTYPE_SIZE * offset)
                        fi.write(struct.pack(self.INDEX_DTYPE, fd.tell()))
                    else:
                        _cur_fd = fd.tell()
                        prev_index = self.NA_INDEX
                        while cur_index != self.NA_INDEX:
                            fd.seek(cur_index + self.DATA_DTYPE_SIZE - self.INDEX_DTYPE_SIZE)
                            prev_index = cur_index
                            (cur_index,) = struct.unpack(self.INDEX_DTYPE, fd.read(self.INDEX_DTYPE_SIZE))
                        fd.seek(prev_index + self.DATA_DTYPE_SIZE - self.INDEX_DTYPE_SIZE)
                        fd.write(struct.pack(self.INDEX_DTYPE, _cur_fd))
                        fd.seek(_cur_fd)
                    fd.write(struct.pack(self.DATA_DTYPE, row.date, row.period, row.value, self.NA_INDEX))

    def dump(self, interval='quarterly', overwrite=False):
        if False:
            while True:
                i = 10
        logger.info('start dump pit data......')
        _dump_func = partial(self._dump_pit, interval=interval, overwrite=overwrite)
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.csv_files):
                    p_bar.update()

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        self.dump()
if __name__ == '__main__':
    fire.Fire(DumpPitData)
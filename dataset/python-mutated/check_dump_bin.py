from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import qlib
from qlib.data import D
import fire
import datacompy
import pandas as pd
from tqdm import tqdm
from loguru import logger

class CheckBin:
    NOT_IN_FEATURES = 'not in features'
    COMPARE_FALSE = 'compare False'
    COMPARE_TRUE = 'compare True'
    COMPARE_ERROR = 'compare error'

    def __init__(self, qlib_dir: str, csv_path: str, check_fields: str=None, freq: str='day', symbol_field_name: str='symbol', date_field_name: str='date', file_suffix: str='.csv', max_workers: int=16):
        if False:
            while True:
                i = 10
        '\n\n        Parameters\n        ----------\n        qlib_dir : str\n            qlib dir\n        csv_path : str\n            origin csv path\n        check_fields : str, optional\n            check fields, by default None, check qlib_dir/features/<first_dir>/*.<freq>.bin\n        freq : str, optional\n            freq, value from ["day", "1m"]\n        symbol_field_name: str, optional\n            symbol field name, by default "symbol"\n        date_field_name: str, optional\n            date field name, by default "date"\n        file_suffix: str, optional\n            csv file suffix, by default ".csv"\n        max_workers: int, optional\n            max workers, by default 16\n        '
        self.qlib_dir = Path(qlib_dir).expanduser()
        bin_path_list = list(self.qlib_dir.joinpath('features').iterdir())
        self.qlib_symbols = sorted(map(lambda x: x.name.lower(), bin_path_list))
        qlib.init(provider_uri=str(self.qlib_dir.resolve()), mount_path=str(self.qlib_dir.resolve()), auto_mount=False, redis_port=-1)
        csv_path = Path(csv_path).expanduser()
        self.csv_files = sorted(csv_path.glob(f'*{file_suffix}') if csv_path.is_dir() else [csv_path])
        if check_fields is None:
            check_fields = list(map(lambda x: x.name.split('.')[0], bin_path_list[0].glob(f'*.bin')))
        else:
            check_fields = check_fields.split(',') if isinstance(check_fields, str) else check_fields
        self.check_fields = list(map(lambda x: x.strip(), check_fields))
        self.qlib_fields = list(map(lambda x: f'${x}', self.check_fields))
        self.max_workers = max_workers
        self.symbol_field_name = symbol_field_name
        self.date_field_name = date_field_name
        self.freq = freq
        self.file_suffix = file_suffix

    def _compare(self, file_path: Path):
        if False:
            return 10
        symbol = file_path.name.strip(self.file_suffix)
        if symbol.lower() not in self.qlib_symbols:
            return self.NOT_IN_FEATURES
        qlib_df = D.features([symbol], self.qlib_fields, freq=self.freq)
        qlib_df.rename(columns={_c: _c.strip('$') for _c in qlib_df.columns}, inplace=True)
        origin_df = pd.read_csv(file_path)
        origin_df[self.date_field_name] = pd.to_datetime(origin_df[self.date_field_name])
        if self.symbol_field_name not in origin_df.columns:
            origin_df[self.symbol_field_name] = symbol
        origin_df.set_index([self.symbol_field_name, self.date_field_name], inplace=True)
        origin_df.index.names = qlib_df.index.names
        origin_df = origin_df.reindex(qlib_df.index)
        try:
            compare = datacompy.Compare(origin_df, qlib_df, on_index=True, abs_tol=1e-08, rel_tol=1e-05, df1_name='Original', df2_name='New')
            _r = compare.matches(ignore_extra_columns=True)
            return self.COMPARE_TRUE if _r else self.COMPARE_FALSE
        except Exception as e:
            logger.warning(f'{symbol} compare error: {e}')
            return self.COMPARE_ERROR

    def check(self):
        if False:
            while True:
                i = 10
        'Check whether the bin file after ``dump_bin.py`` is executed is consistent with the original csv file data'
        logger.info('start check......')
        error_list = []
        not_in_features = []
        compare_false = []
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for (file_path, _check_res) in zip(self.csv_files, executor.map(self._compare, self.csv_files)):
                    symbol = file_path.name.strip(self.file_suffix)
                    if _check_res == self.NOT_IN_FEATURES:
                        not_in_features.append(symbol)
                    elif _check_res == self.COMPARE_ERROR:
                        error_list.append(symbol)
                    elif _check_res == self.COMPARE_FALSE:
                        compare_false.append(symbol)
                    p_bar.update()
        logger.info('end of check......')
        if error_list:
            logger.warning(f'compare error: {error_list}')
        if not_in_features:
            logger.warning(f'not in features: {not_in_features}')
        if compare_false:
            logger.warning(f'compare False: {compare_false}')
        logger.info(f'total {len(self.csv_files)}, {len(error_list)} errors, {len(not_in_features)} not in features, {len(compare_false)} compare false')
if __name__ == '__main__':
    fire.Fire(CheckBin)
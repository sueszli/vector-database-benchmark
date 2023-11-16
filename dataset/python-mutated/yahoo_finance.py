import os
import pandas as pd
from typing import Optional, Union
from .base import YahooFinanceConnectorConfig, BaseConnector
import time
from ..helpers.path import find_project_root
from ..constants import DEFAULT_FILE_PERMISSIONS
import hashlib

class YahooFinanceConnector(BaseConnector):
    """
    Yahoo Finance connector for retrieving stock data.
    """
    _cache_interval: int = 600

    def __init__(self, stock_ticker: Optional[str]=None, config: Optional[Union[YahooFinanceConnectorConfig, dict]]=None, cache_interval: int=600):
        if False:
            for i in range(10):
                print('nop')
        if not stock_ticker and (not config):
            raise ValueError('You must specify either a stock ticker or a config object.')
        try:
            import yfinance
        except ImportError as e:
            raise ImportError('Could not import yfinance python package. Please install it with `pip install yfinance`.') from e
        if not isinstance(config, YahooFinanceConnectorConfig):
            if not config:
                config = {}
            if stock_ticker:
                config['table'] = stock_ticker
            yahoo_finance_config = YahooFinanceConnectorConfig(**config)
        else:
            yahoo_finance_config = config
        self._cache_interval = cache_interval
        super().__init__(yahoo_finance_config)
        self.ticker = yfinance.Ticker(self._config.table)

    def head(self):
        if False:
            return 10
        '\n        Return the head of the data source that the connector is connected to.\n\n        Returns:\n            DataFrameType: The head of the data source that the connector is\n            connected to.\n        '
        return self.ticker.history(period='5d')

    def _get_cache_path(self, include_additional_filters: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the path of the cache file.\n\n        Returns:\n            str: The path of the cache file.\n        '
        cache_dir = os.path.join(os.getcwd(), '')
        try:
            cache_dir = os.path.join(find_project_root(), 'cache')
        except ValueError:
            cache_dir = os.path.join(os.getcwd(), 'cache')
        return os.path.join(cache_dir, f'{self._config.table}_data.parquet')

    def _get_cache_path(self):
        if False:
            print('Hello World!')
        '\n        Return the path of the cache file for Yahoo Finance data.\n        '
        try:
            cache_dir = os.path.join(find_project_root(), 'cache')
        except ValueError:
            cache_dir = os.path.join(os.getcwd(), 'cache')
        os.makedirs(cache_dir, mode=DEFAULT_FILE_PERMISSIONS, exist_ok=True)
        return os.path.join(cache_dir, f'{self._config.table}_data.parquet')

    def _cached(self):
        if False:
            while True:
                i = 10
        '\n        Return the cached Yahoo Finance data if it exists and is not older than the\n        cache interval.\n\n        Returns:\n            DataFrame|None: The cached data if it exists and is not older than the cache\n            interval, None otherwise.\n        '
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return None
        if os.path.getmtime(cache_path) < time.time() - self._cache_interval:
            if self.logger:
                self.logger.log(f'Deleting expired cached data from {cache_path}')
            os.remove(cache_path)
            return None
        if self.logger:
            self.logger.log(f'Loading cached data from {cache_path}')
        return cache_path

    def execute(self):
        if False:
            print('Hello World!')
        '\n        Execute the connector and return the result.\n\n        Returns:\n            DataFrameType: The result of the connector.\n        '
        if (cached_path := self._cached()):
            return pd.read_parquet(cached_path)
        stock_data = self.ticker.history(period='max')
        stock_data.to_parquet(self._get_cache_path())
        return stock_data

    @property
    def rows_count(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the number of rows in the data source that the connector is\n        connected to.\n\n        Returns:\n            int: The number of rows in the data source that the connector is\n            connected to.\n        '
        stock_data = self.execute()
        return len(stock_data)

    @property
    def columns_count(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the number of columns in the data source that the connector is\n        connected to.\n\n        Returns:\n            int: The number of columns in the data source that the connector is\n            connected to.\n        '
        stock_data = self.execute()
        return len(stock_data.columns)

    @property
    def column_hash(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the hash code that is unique to the columns of the data source\n        that the connector is connected to.\n\n        Returns:\n            int: The hash code that is unique to the columns of the data source\n            that the connector is connected to.\n        '
        stock_data = self.execute()
        columns_str = '|'.join(stock_data.columns)
        return hashlib.sha256(columns_str.encode('utf-8')).hexdigest()

    @property
    def fallback_name(self):
        if False:
            print('Hello World!')
        '\n        Return the fallback name of the connector.\n\n        Returns:\n            str: The fallback name of the connector.\n        '
        return self._config.table
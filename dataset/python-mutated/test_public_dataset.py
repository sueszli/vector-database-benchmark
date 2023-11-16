import os
import pandas as pd
import pytest
from unittest import TestCase
from bigdl.chronos.data.utils.public_dataset import PublicDataset
from ... import op_torch, op_tf2

@op_torch
@op_tf2
class TestPublicDataset(TestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        pass

    def teardown_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_init_get_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        name = 'nyc_taxi'
        path = '~/.chronos/dataset/'
        with pytest.raises(RuntimeError):
            PublicDataset(name, path, redownload=False).get_public_data(chunk_size='1024')

    def test_get_nyc_taxi(self):
        if False:
            print('Hello World!')
        name = 'nyc_taxi'
        path = '~/.chronos/dataset'
        if os.environ.get('FTP_URI', None):
            file_url = f"{os.getenv('FTP_URI')}/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv"
            public_data = PublicDataset(name, path, redownload=False, with_split=False)
            public_data.df = pd.read_csv(file_url, parse_dates=['timestamp'])
            tsdata = public_data.get_tsdata(target_col='value', dt_col='timestamp')
            assert set(tsdata.df.columns) == {'id', 'timestamp', 'value'}
            assert tsdata.df.shape == (10320, 3)
            tsdata._check_basic_invariants()

    def test_get_network_traffic(self):
        if False:
            for i in range(10):
                print('nop')
        name = 'network_traffic'
        path = '~/.chronos/dataset'
        if os.environ.get('FTP_URI', None):
            file_url = f"{os.getenv('FTP_URI')}/analytics-zoo-data/network-traffic/data/data.csv"
            public_data = PublicDataset(name, path, redownload=False, with_split=False)
            public_data.df = pd.read_csv(file_url)
            public_data.df.StartTime = pd.to_datetime(public_data.df.StartTime)
            public_data.df.AvgRate = public_data.df.AvgRate.apply(lambda x: float(x[:-4]) if x.endswith('Mbps') else float(x[:-4]) * 1000)
            tsdata = public_data.get_tsdata(target_col=['AvgRate', 'total'], dt_col='StartTime', repair=False)
            assert tsdata.df.shape == (8760, 5)
            assert set(tsdata.df.columns) == {'StartTime', 'EndTime', 'AvgRate', 'total', 'id'}
            tsdata._check_basic_invariants()

    def test_get_fsi(self):
        if False:
            i = 10
            return i + 15
        name = 'fsi'
        path = '~/.chronos/dataset'
        if os.environ.get('FTP_URI', None):
            file_url = f"{os.getenv('FTP_URI')}/analytics-zoo-data/chronos-aiops/m_1932.csv"
            public_data = PublicDataset(name, path, redownload=False, with_split=False)
            public_data.df = pd.read_csv(file_url, usecols=[1, 2, 3], names=['time_step', 'cpu_usage', 'mem_usage'])
            public_data.df.sort_values(by='time_step', inplace=True)
            public_data.df.reset_index(inplace=True, drop=True)
            public_data.df.time_step = pd.to_datetime(public_data.df.time_step, unit='s', origin=pd.Timestamp('2018-01-01'))
            tsdata = public_data.get_tsdata(dt_col='time_step', target_col='cpu_usage', repair=False)
            assert tsdata.df.shape == (61570, 4)
            assert set(tsdata.df.columns) == {'time_step', 'cpu_usage', 'mem_usage', 'id'}
            tsdata._check_basic_invariants()

    def test_get_uci_electricity(self):
        if False:
            while True:
                i = 10
        name = 'uci_electricity'
        path = '~/.chronos/dataset'
        if os.environ.get('FTP_URI', None):
            file_url = f"{os.getenv('FTP_URI')}/analytics-zoo-data/apps/ElectricityLD/uci_electricity_data.csv"
            public_data = PublicDataset(name, path, redownload=False, with_split=False)
            df = pd.read_csv(file_url, delimiter=';', parse_dates=['Unnamed: 0'], nrows=10000, low_memory=False)
            public_data.df = pd.melt(df, id_vars=['Unnamed: 0'], value_vars=df.T.index[1:]).rename(columns={'Unnamed: 0': 'timestamp', 'variable': 'id'})
            tsdata = public_data.get_tsdata(dt_col='timestamp', target_col='value', id_col='id')
            assert set(tsdata.df.columns) == {'id', 'timestamp', 'value'}
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sasreader import read_sas

def numeric_as_float(data):
    if False:
        return 10
    for v in data.columns:
        if data[v].dtype is np.dtype('int64'):
            data[v] = data[v].astype(np.float64)

class TestXport:

    @pytest.fixture
    def file01(self, datapath):
        if False:
            for i in range(10):
                print('nop')
        return datapath('io', 'sas', 'data', 'DEMO_G.xpt')

    @pytest.fixture
    def file02(self, datapath):
        if False:
            i = 10
            return i + 15
        return datapath('io', 'sas', 'data', 'SSHSV1_A.xpt')

    @pytest.fixture
    def file03(self, datapath):
        if False:
            i = 10
            return i + 15
        return datapath('io', 'sas', 'data', 'DRXFCD_G.xpt')

    @pytest.fixture
    def file04(self, datapath):
        if False:
            for i in range(10):
                print('nop')
        return datapath('io', 'sas', 'data', 'paxraw_d_short.xpt')

    @pytest.fixture
    def file05(self, datapath):
        if False:
            while True:
                i = 10
        return datapath('io', 'sas', 'data', 'DEMO_PUF.cpt')

    @pytest.mark.slow
    def test1_basic(self, file01):
        if False:
            print('Hello World!')
        data_csv = pd.read_csv(file01.replace('.xpt', '.csv'))
        numeric_as_float(data_csv)
        data = read_sas(file01, format='xport')
        tm.assert_frame_equal(data, data_csv)
        num_rows = data.shape[0]
        with read_sas(file01, format='xport', iterator=True) as reader:
            data = reader.read(num_rows + 100)
        assert data.shape[0] == num_rows
        with read_sas(file01, format='xport', iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])
        with read_sas(file01, format='xport', chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :])
        m = 0
        with read_sas(file01, format='xport', chunksize=100) as reader:
            for x in reader:
                m += x.shape[0]
        assert m == num_rows
        data = read_sas(file01)
        tm.assert_frame_equal(data, data_csv)

    def test1_index(self, file01):
        if False:
            print('Hello World!')
        data_csv = pd.read_csv(file01.replace('.xpt', '.csv'))
        data_csv = data_csv.set_index('SEQN')
        numeric_as_float(data_csv)
        data = read_sas(file01, index='SEQN', format='xport')
        tm.assert_frame_equal(data, data_csv, check_index_type=False)
        with read_sas(file01, index='SEQN', format='xport', iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)
        with read_sas(file01, index='SEQN', format='xport', chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

    def test1_incremental(self, file01):
        if False:
            print('Hello World!')
        data_csv = pd.read_csv(file01.replace('.xpt', '.csv'))
        data_csv = data_csv.set_index('SEQN')
        numeric_as_float(data_csv)
        with read_sas(file01, index='SEQN', chunksize=1000) as reader:
            all_data = list(reader)
        data = pd.concat(all_data, axis=0)
        tm.assert_frame_equal(data, data_csv, check_index_type=False)

    def test2(self, file02):
        if False:
            print('Hello World!')
        data_csv = pd.read_csv(file02.replace('.xpt', '.csv'))
        numeric_as_float(data_csv)
        data = read_sas(file02)
        tm.assert_frame_equal(data, data_csv)

    def test2_binary(self, file02):
        if False:
            for i in range(10):
                print('nop')
        data_csv = pd.read_csv(file02.replace('.xpt', '.csv'))
        numeric_as_float(data_csv)
        with open(file02, 'rb') as fd:
            data = read_sas(fd, format='xport')
        tm.assert_frame_equal(data, data_csv)

    def test_multiple_types(self, file03):
        if False:
            return 10
        data_csv = pd.read_csv(file03.replace('.xpt', '.csv'))
        data = read_sas(file03, encoding='utf-8')
        tm.assert_frame_equal(data, data_csv)

    def test_truncated_float_support(self, file04):
        if False:
            return 10
        data_csv = pd.read_csv(file04.replace('.xpt', '.csv'))
        data = read_sas(file04, format='xport')
        tm.assert_frame_equal(data.astype('int64'), data_csv)

    def test_cport_header_found_raises(self, file05):
        if False:
            i = 10
            return i + 15
        msg = 'Header record indicates a CPORT file, which is not readable.'
        with pytest.raises(ValueError, match=msg):
            read_sas(file05, format='xport')
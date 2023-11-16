import glob
import os
from datetime import datetime
import vaex

def test_open_two_big_csv_convert():
    if False:
        return 10
    big_and_biggest_csv = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1*.csv'
    os.remove('/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1_01.csv.hdf5')
    os.remove('/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1.csv.hdf5')
    os.remove('/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1.csv_and_1_more.hdf5')
    start = datetime.now()
    df = vaex.open(big_and_biggest_csv, convert=True)
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(duration, df.length(), int(df.length() / duration.total_seconds())))

def test_open_several_medium_csv_convert():
    if False:
        for i in range(10):
            print('nop')
    csv_glob = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-01_*.csv'
    for path in glob.glob(csv_glob):
        os.remove(path + '.hdf5')
    os.remove('/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-01_0.csv_and_3_more.hdf5')
    start = datetime.now()
    df = vaex.open(csv_glob, convert=True)
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(duration, df.length(), int(df.length() / duration.total_seconds())))
    assert df.length() == 3999999

def test_from_big_csv_read():
    if False:
        for i in range(10):
            print('nop')
    csv = '/Users/byaminov/fun/datasets/yellow_tripdata_2019-01.csv'
    start = datetime.now()
    read_length = 0
    read_length += len(vaex.from_csv(csv))
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(duration, read_length, int(read_length / duration.total_seconds())))
    assert read_length == 7667792

def test_from_big_csv_convert():
    if False:
        while True:
            i = 10
    csv = '/Users/byaminov/fun/datasets/yellow_tripdata_2019-01.csv'
    os.remove(csv + '.hdf5')
    start = datetime.now()
    df = vaex.from_csv(csv, convert=True)
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(duration, df.length(), int(df.length() / duration.total_seconds())))
    assert df.length() == 7667792

def test_read_csv_and_convert():
    if False:
        i = 10
        return i + 15
    test_path = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-01_*.csv'
    import os
    import glob
    for hdf_file in glob.glob(test_path.replace('.csv', '.hdf5')):
        print('deleting %s' % hdf_file)
        os.remove(hdf_file)
    start = datetime.now()
    df = vaex.read_csv_and_convert(test_path, copy_index=False)
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(duration, df.length(), df.length() / duration.total_seconds()))
    assert df.length() == 3999999

def test_pandas_read_csv_chunked():
    if False:
        for i in range(10):
            print('nop')
    test_path = '/Users/byaminov/fun/datasets/yellow_tripdata_2019-01.csv'
    import pandas as pd
    start = datetime.now()
    n_read = 0
    for df in pd.read_csv(test_path, chunksize=1000000):
        n_read += len(df)
    duration = datetime.now() - start
    print('it took {} to read {:,} rows, which is {:,} rows per second'.format(duration, n_read, int(n_read / duration.total_seconds())))
    assert n_read == 7667792

def test_arrow_read_csv_chunked():
    if False:
        return 10
    test_path = '/Users/byaminov/fun/datasets/yellow_tripdata_2019-01.csv'
    from pyarrow import csv
    start = datetime.now()
    table = csv.read_csv(test_path)
    duration = datetime.now() - start
    print('it took {} to read {:,} rows, which is {:,} rows per second'.format(duration, len(table), int(len(table) / duration.total_seconds())))
    assert len(table) == 7667792
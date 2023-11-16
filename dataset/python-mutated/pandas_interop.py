from __future__ import annotations
import contextlib
import csv
import os
import tempfile
from posixpath import join as pjoin
import ibis.common.exceptions as com
import ibis.expr.schema as sch
from ibis import util
from ibis.config import options

class DataFrameWriter:
    """Interface class for writing pandas objects to Impala tables.

    Notes
    -----
    Class takes ownership of any temporary data written to HDFS
    """

    def __init__(self, client, df):
        if False:
            for i in range(10):
                print('nop')
        self.client = client
        self.df = df
        self.temp_hdfs_dirs = set()

    def write_temp_csv(self):
        if False:
            i = 10
            return i + 15
        temp_hdfs_dir = pjoin(options.impala.temp_hdfs_path, f'pandas_{util.guid()}')
        self.client.hdfs.mkdir(temp_hdfs_dir)
        self.temp_hdfs_dirs.add(temp_hdfs_dir)
        hdfs_path = pjoin(temp_hdfs_dir, '0.csv')
        self.write_csv(hdfs_path)
        return temp_hdfs_dir

    def write_csv(self, path):
        if False:
            i = 10
            return i + 15
        with tempfile.TemporaryDirectory() as f:
            tmp_file_path = os.path.join(f, 'impala_temp_file.csv')
            if options.verbose:
                util.log(f'Writing DataFrame to temporary directory {tmp_file_path}')
            self.df.to_csv(tmp_file_path, header=False, index=False, sep=',', quoting=csv.QUOTE_NONE, escapechar='\\', na_rep='#NULL')
            if options.verbose:
                util.log(f'Writing CSV to: {path}')
            self.client.hdfs.put(tmp_file_path, path)
        return path

    def get_schema(self):
        if False:
            while True:
                i = 10
        return sch.infer(self.df)

    def delimited_table(self, csv_dir, database=None):
        if False:
            return 10
        return self.client.delimited_file(csv_dir, self.get_schema(), name=f'ibis_tmp_pandas_{util.guid()}', database=database, delimiter=',', na_rep='#NULL', escapechar='\\\\', external=True, persist=False)

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        with contextlib.suppress(com.IbisError):
            self.cleanup()
        return False

    def cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        while self.temp_hdfs_dirs:
            self.client.hdfs.rm(self.temp_hdfs_dirs.pop(), recursive=True)
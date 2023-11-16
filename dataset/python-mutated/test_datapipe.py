import copy
import itertools
import os
import os.path
import pickle
import pydoc
import random
import sys
import tempfile
import warnings
from functools import partial
from typing import Any, Awaitable, Dict, Generic, Iterator, List, Optional, Set, Tuple, Type, TypeVar, Union, TYPE_CHECKING
if not TYPE_CHECKING:
    from typing_extensions import NamedTuple
else:
    from typing import NamedTuple
from unittest import skipIf
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.datapipes as dp
import torch.utils.data.graph
import torch.utils.data.graph_settings
from torch.testing._internal.common_utils import TestCase, run_tests, suppress_warnings, skipIfTorchDynamo
from torch.utils.data import DataLoader, DataChunk, IterDataPipe, MapDataPipe, RandomSampler, argument_validation, runtime_validation, runtime_validation_disabled
from torch.utils.data.graph import traverse_dps
from torch.utils.data.datapipes.utils.common import StreamWrapper
from torch.utils.data.datapipes.utils.decoder import basichandlers as decoder_basichandlers
from torch.utils.data.datapipes.utils.snapshot import _simple_graph_snapshot_restoration
from torch.utils.data.datapipes.dataframe import CaptureDataFrame
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
try:
    import dill
    dill.extend(use_dill=False)
    HAS_DILL = True
except ImportError:
    HAS_DILL = False
skipIfNoDill = skipIf(not HAS_DILL, 'no dill')
try:
    import pandas
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
skipIfNoDataFrames = skipIf(not HAS_PANDAS, 'no dataframes (pandas)')
skipTyping = skipIf(True, 'TODO: Fix typing bug')
T_co = TypeVar('T_co', covariant=True)

def create_temp_dir_and_files():
    if False:
        return 10
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, suffix='.txt') as f:
        temp_file1_name = f.name
    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, suffix='.byte') as f:
        temp_file2_name = f.name
    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, suffix='.empty') as f:
        temp_file3_name = f.name
    with open(temp_file1_name, 'w') as f1:
        f1.write('0123456789abcdef')
    with open(temp_file2_name, 'wb') as f2:
        f2.write(b'0123456789abcdef')
    temp_sub_dir = tempfile.TemporaryDirectory(dir=temp_dir_path)
    temp_sub_dir_path = temp_sub_dir.name
    with tempfile.NamedTemporaryFile(dir=temp_sub_dir_path, delete=False, suffix='.txt') as f:
        temp_sub_file1_name = f.name
    with tempfile.NamedTemporaryFile(dir=temp_sub_dir_path, delete=False, suffix='.byte') as f:
        temp_sub_file2_name = f.name
    with open(temp_sub_file1_name, 'w') as f1:
        f1.write('0123456789abcdef')
    with open(temp_sub_file2_name, 'wb') as f2:
        f2.write(b'0123456789abcdef')
    return [(temp_dir, temp_file1_name, temp_file2_name, temp_file3_name), (temp_sub_dir, temp_sub_file1_name, temp_sub_file2_name)]

def reset_after_n_next_calls(datapipe: Union[IterDataPipe[T_co], MapDataPipe[T_co]], n: int) -> Tuple[List[T_co], List[T_co]]:
    if False:
        i = 10
        return i + 15
    '\n    Given a DataPipe and integer n, iterate the DataPipe for n elements and store the elements into a list\n    Then, reset the DataPipe and return a tuple of two lists\n        1. A list of elements yielded before the reset\n        2. A list of all elements of the DataPipe after the reset\n    '
    it = iter(datapipe)
    res_before_reset = []
    for _ in range(n):
        res_before_reset.append(next(it))
    return (res_before_reset, list(datapipe))

def odd_or_even(x: int) -> int:
    if False:
        i = 10
        return i + 15
    return x % 2

class TestDataChunk(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.elements = list(range(10))
        random.shuffle(self.elements)
        self.chunk: DataChunk[int] = DataChunk(self.elements)

    def test_getitem(self):
        if False:
            i = 10
            return i + 15
        for i in range(10):
            self.assertEqual(self.elements[i], self.chunk[i])

    def test_iter(self):
        if False:
            return 10
        for (ele, dc) in zip(self.elements, iter(self.chunk)):
            self.assertEqual(ele, dc)

    def test_len(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(self.elements), len(self.chunk))

    def test_as_string(self):
        if False:
            return 10
        self.assertEqual(str(self.chunk), str(self.elements))
        batch = [self.elements] * 3
        chunks: List[DataChunk[int]] = [DataChunk(self.elements)] * 3
        self.assertEqual(str(batch), str(chunks))

    def test_sort(self):
        if False:
            while True:
                i = 10
        chunk: DataChunk[int] = DataChunk(self.elements)
        chunk.sort()
        self.assertTrue(isinstance(chunk, DataChunk))
        for (i, d) in enumerate(chunk):
            self.assertEqual(i, d)

    def test_reverse(self):
        if False:
            for i in range(10):
                print('nop')
        chunk: DataChunk[int] = DataChunk(self.elements)
        chunk.reverse()
        self.assertTrue(isinstance(chunk, DataChunk))
        for i in range(10):
            self.assertEqual(chunk[i], self.elements[9 - i])

    def test_random_shuffle(self):
        if False:
            for i in range(10):
                print('nop')
        elements = list(range(10))
        chunk: DataChunk[int] = DataChunk(elements)
        rng = random.Random(0)
        rng.shuffle(chunk)
        rng = random.Random(0)
        rng.shuffle(elements)
        self.assertEqual(chunk, elements)

class TestStreamWrapper(TestCase):

    class _FakeFD:

        def __init__(self, filepath):
            if False:
                print('Hello World!')
            self.filepath = filepath
            self.opened = False
            self.closed = False

        def open(self):
            if False:
                return 10
            self.opened = True

        def read(self):
            if False:
                while True:
                    i = 10
            if self.opened:
                return ''.join(self)
            else:
                raise OSError('Cannot read from un-opened file descriptor')

        def __iter__(self):
            if False:
                print('Hello World!')
            for i in range(5):
                yield str(i)

        def close(self):
            if False:
                print('Hello World!')
            if self.opened:
                self.opened = False
                self.closed = True

        def __repr__(self):
            if False:
                print('Hello World!')
            return 'FakeFD'

    def test_dir(self):
        if False:
            while True:
                i = 10
        fd = TestStreamWrapper._FakeFD('')
        wrap_fd = StreamWrapper(fd)
        s = set(dir(wrap_fd))
        for api in ['open', 'read', 'close']:
            self.assertTrue(api in s)

    @skipIfTorchDynamo
    def test_api(self):
        if False:
            while True:
                i = 10
        fd = TestStreamWrapper._FakeFD('')
        wrap_fd = StreamWrapper(fd)
        self.assertFalse(fd.opened)
        self.assertFalse(fd.closed)
        with self.assertRaisesRegex(IOError, 'Cannot read from'):
            wrap_fd.read()
        wrap_fd.open()
        self.assertTrue(fd.opened)
        self.assertEqual('01234', wrap_fd.read())
        del wrap_fd
        self.assertFalse(fd.opened)
        self.assertTrue(fd.closed)

    def test_pickle(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryFile() as f:
            with self.assertRaises(TypeError) as ctx1:
                pickle.dumps(f)
            wrap_f = StreamWrapper(f)
            with self.assertRaises(TypeError) as ctx2:
                pickle.dumps(wrap_f)
            self.assertEqual(str(ctx1.exception), str(ctx2.exception))
        fd = TestStreamWrapper._FakeFD('')
        wrap_fd = StreamWrapper(fd)
        _ = pickle.loads(pickle.dumps(wrap_fd))

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        fd = TestStreamWrapper._FakeFD('')
        wrap_fd = StreamWrapper(fd)
        self.assertEqual(str(wrap_fd), 'StreamWrapper<FakeFD>')
        with tempfile.TemporaryFile() as f:
            wrap_f = StreamWrapper(f)
            self.assertEqual(str(wrap_f), 'StreamWrapper<' + str(f) + '>')

class TestIterableDataPipeBasic(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        ret = create_temp_dir_and_files()
        self.temp_dir = ret[0][0]
        self.temp_files = ret[0][1:]
        self.temp_sub_dir = ret[1][0]
        self.temp_sub_files = ret[1][1:]

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        try:
            self.temp_sub_dir.cleanup()
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn(f'TestIterableDatasetBasic was not able to cleanup temp dir due to {str(e)}')

    def test_listdirfiles_iterable_datapipe(self):
        if False:
            for i in range(10):
                print('nop')
        temp_dir = self.temp_dir.name
        datapipe: IterDataPipe = dp.iter.FileLister(temp_dir, '')
        count = 0
        for pathname in datapipe:
            count = count + 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, len(self.temp_files))
        count = 0
        datapipe = dp.iter.FileLister(temp_dir, '', recursive=True)
        for pathname in datapipe:
            count = count + 1
            self.assertTrue(pathname in self.temp_files or pathname in self.temp_sub_files)
        self.assertEqual(count, len(self.temp_files) + len(self.temp_sub_files))
        temp_files = self.temp_files
        datapipe = dp.iter.FileLister([temp_dir, *temp_files])
        count = 0
        for pathname in datapipe:
            count += 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, 2 * len(self.temp_files))
        datapipe = datapipe.list_files()
        count = 0
        for pathname in datapipe:
            count += 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, 2 * len(self.temp_files))

    def test_listdirfilesdeterministic_iterable_datapipe(self):
        if False:
            return 10
        temp_dir = self.temp_dir.name
        datapipe = dp.iter.FileLister(temp_dir, '')
        self.assertEqual(list(datapipe), list(datapipe))
        datapipe = dp.iter.FileLister(temp_dir, '', recursive=True)
        self.assertEqual(list(datapipe), list(datapipe))

    def test_openfilesfromdisk_iterable_datapipe(self):
        if False:
            while True:
                i = 10
        from torch.utils.data.datapipes.iter import FileLister, FileOpener
        temp_dir = self.temp_dir.name
        datapipe1 = FileLister(temp_dir, '')
        datapipe2 = FileOpener(datapipe1, mode='b')
        count = 0
        for rec in datapipe2:
            count = count + 1
            self.assertTrue(rec[0] in self.temp_files)
            with open(rec[0], 'rb') as f:
                self.assertEqual(rec[1].read(), f.read())
                rec[1].close()
        self.assertEqual(count, len(self.temp_files))
        datapipe3 = datapipe1.open_files(mode='b')
        count = 0
        for rec in datapipe3:
            count = count + 1
            self.assertTrue(rec[0] in self.temp_files)
            with open(rec[0], 'rb') as f:
                self.assertEqual(rec[1].read(), f.read())
                rec[1].close()
        self.assertEqual(count, len(self.temp_files))
        with self.assertRaises(TypeError):
            len(datapipe3)

    def test_routeddecoder_iterable_datapipe(self):
        if False:
            for i in range(10):
                print('nop')
        temp_dir = self.temp_dir.name
        temp_pngfile_pathname = os.path.join(temp_dir, 'test_png.png')
        png_data = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.single)
        np.save(temp_pngfile_pathname, png_data)
        datapipe1 = dp.iter.FileLister(temp_dir, ['*.png', '*.txt'])
        datapipe2 = dp.iter.FileOpener(datapipe1, mode='b')

        def _png_decoder(extension, data):
            if False:
                return 10
            if extension != 'png':
                return None
            return np.load(data)

        def _helper(prior_dp, dp, channel_first=False):
            if False:
                return 10
            for inp in prior_dp:
                self.assertFalse(inp[1].closed)
            for (inp, rec) in zip(prior_dp, dp):
                ext = os.path.splitext(rec[0])[1]
                if ext == '.png':
                    expected = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.single)
                    if channel_first:
                        expected = expected.transpose(2, 0, 1)
                    self.assertEqual(rec[1], expected)
                else:
                    with open(rec[0], 'rb') as f:
                        self.assertEqual(rec[1], f.read().decode('utf-8'))
                self.assertTrue(inp[1].closed)
        cached = list(datapipe2)
        with warnings.catch_warnings(record=True) as wa:
            datapipe3 = dp.iter.RoutedDecoder(cached, _png_decoder)
        datapipe3.add_handler(decoder_basichandlers)
        _helper(cached, datapipe3)
        cached = list(datapipe2)
        with warnings.catch_warnings(record=True) as wa:
            datapipe4 = dp.iter.RoutedDecoder(cached, decoder_basichandlers)
        datapipe4.add_handler(_png_decoder)
        _helper(cached, datapipe4, channel_first=True)

    def test_groupby_iterable_datapipe(self):
        if False:
            for i in range(10):
                print('nop')
        file_list = ['a.png', 'b.png', 'c.json', 'a.json', 'c.png', 'b.json', 'd.png', 'd.json', 'e.png', 'f.json', 'g.png', 'f.png', 'g.json', 'e.json', 'h.txt', 'h.json']
        import io
        datapipe1 = dp.iter.IterableWrapper([(filename, io.BytesIO(b'12345abcde')) for filename in file_list])

        def group_fn(data):
            if False:
                while True:
                    i = 10
            (filepath, _) = data
            return os.path.basename(filepath).split('.')[0]
        datapipe2 = dp.iter.Grouper(datapipe1, group_key_fn=group_fn, group_size=2)

        def order_fn(data):
            if False:
                for i in range(10):
                    print('nop')
            data.sort(key=lambda f: f[0], reverse=True)
            return data
        datapipe3 = dp.iter.Mapper(datapipe2, fn=order_fn)
        expected_result = [('a.png', 'a.json'), ('c.png', 'c.json'), ('b.png', 'b.json'), ('d.png', 'd.json'), ('f.png', 'f.json'), ('g.png', 'g.json'), ('e.png', 'e.json'), ('h.txt', 'h.json')]
        count = 0
        for (rec, expected) in zip(datapipe3, expected_result):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0][0]), expected[0])
            self.assertEqual(os.path.basename(rec[1][0]), expected[1])
            for i in [0, 1]:
                self.assertEqual(rec[i][1].read(), b'12345abcde')
                rec[i][1].close()
        self.assertEqual(count, 8)
        datapipe4 = dp.iter.Grouper(datapipe1, group_key_fn=group_fn, keep_key=True, group_size=2)

        def order_fn(data):
            if False:
                for i in range(10):
                    print('nop')
            data[1].sort(key=lambda f: f[0], reverse=True)
            return data
        datapipe5 = dp.iter.Mapper(datapipe4, fn=order_fn)
        expected_result = [('a', ('a.png', 'a.json')), ('c', ('c.png', 'c.json')), ('b', ('b.png', 'b.json')), ('d', ('d.png', 'd.json')), ('f', ('f.png', 'f.json')), ('g', ('g.png', 'g.json')), ('e', ('e.png', 'e.json')), ('h', ('h.txt', 'h.json'))]
        count = 0
        for (rec, expected) in zip(datapipe5, expected_result):
            count = count + 1
            self.assertEqual(rec[0], expected[0])
            self.assertEqual(rec[1][0][0], expected[1][0])
            self.assertEqual(rec[1][1][0], expected[1][1])
            for i in [0, 1]:
                self.assertEqual(rec[1][i][1].read(), b'12345abcde')
                rec[1][i][1].close()
        self.assertEqual(count, 8)

    def test_demux_mux_datapipe(self):
        if False:
            for i in range(10):
                print('nop')
        numbers = NumbersDataset(10)
        (n1, n2) = numbers.demux(2, lambda x: x % 2)
        self.assertEqual([0, 2, 4, 6, 8], list(n1))
        self.assertEqual([1, 3, 5, 7, 9], list(n2))
        numbers = NumbersDataset(10)
        (n1, n2, n3) = numbers.demux(3, lambda x: x % 3)
        n = n1.mux(n2, n3)
        self.assertEqual(list(range(9)), list(n))
        source_numbers = list(range(0, 10)) + [10, 12]
        numbers_dp = dp.iter.IterableWrapper(source_numbers)
        (n1, n2) = numbers_dp.demux(2, lambda x: x % 2)
        self.assertEqual([0, 2, 4, 6, 8, 10, 12], list(n1))
        self.assertEqual([1, 3, 5, 7, 9], list(n2))
        n = n1.mux(n2)
        self.assertEqual(list(range(10)), list(n))

    @suppress_warnings
    def test_map_with_col_file_handle_datapipe(self):
        if False:
            while True:
                i = 10
        temp_dir = self.temp_dir.name
        datapipe1 = dp.iter.FileLister(temp_dir, '')
        datapipe2 = dp.iter.FileOpener(datapipe1)

        def _helper(datapipe):
            if False:
                i = 10
                return i + 15
            dp1 = datapipe.map(lambda x: x.read(), input_col=1)
            dp2 = datapipe.map(lambda x: (x[0], x[1].read()))
            self.assertEqual(list(dp1), list(dp2))
        _helper(datapipe2)
        datapipe3 = datapipe2.map(lambda x: list(x))
        _helper(datapipe3)

@skipIfNoDataFrames
class TestCaptureDataFrame(TestCase):

    def get_new_df(self):
        if False:
            print('Hello World!')
        return df_wrapper.create_dataframe([[1, 2]], columns=['a', 'b'])

    def compare_capture_and_eager(self, operations):
        if False:
            while True:
                i = 10
        cdf = CaptureDataFrame()
        cdf = operations(cdf)
        df = self.get_new_df()
        cdf = cdf.apply_ops(df)
        df = self.get_new_df()
        df = operations(df)
        self.assertTrue(df.equals(cdf))

    def test_basic_capture(self):
        if False:
            print('Hello World!')

        def operations(df):
            if False:
                return 10
            df['c'] = df.b + df['a'] * 7
            return df
        self.compare_capture_and_eager(operations)

class TestDataFramesPipes(TestCase):
    """
        Most of test will fail if pandas instaled, but no dill available.
        Need to rework them to avoid multiple skips.
    """

    def _get_datapipe(self, range=10, dataframe_size=7):
        if False:
            for i in range(10):
                print('nop')
        return NumbersDataset(range).map(lambda i: (i, i % 3))

    def _get_dataframes_pipe(self, range=10, dataframe_size=7):
        if False:
            i = 10
            return i + 15
        return NumbersDataset(range).map(lambda i: (i, i % 3))._to_dataframes_pipe(columns=['i', 'j'], dataframe_size=dataframe_size)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_capture(self):
        if False:
            while True:
                i = 10
        dp_numbers = self._get_datapipe().map(lambda x: (x[0], x[1], x[1] + 3 * x[0]))
        df_numbers = self._get_dataframes_pipe()
        df_numbers['k'] = df_numbers['j'] + df_numbers.i * 3
        expected = list(dp_numbers)
        actual = list(df_numbers)
        self.assertEqual(expected, actual)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_shuffle(self):
        if False:
            i = 10
            return i + 15
        df_numbers = self._get_dataframes_pipe(range=1000).shuffle()
        dp_numbers = self._get_datapipe(range=1000)
        df_result = [tuple(item) for item in df_numbers]
        self.assertNotEqual(list(dp_numbers), df_result)
        self.assertEqual(list(dp_numbers), sorted(df_result))

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_batch(self):
        if False:
            for i in range(10):
                print('nop')
        df_numbers = self._get_dataframes_pipe(range=100).batch(8)
        df_numbers_list = list(df_numbers)
        last_batch = df_numbers_list[-1]
        self.assertEqual(4, len(last_batch))
        unpacked_batch = [tuple(row) for row in last_batch]
        self.assertEqual([(96, 0), (97, 1), (98, 2), (99, 0)], unpacked_batch)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_unbatch(self):
        if False:
            return 10
        df_numbers = self._get_dataframes_pipe(range=100).batch(8).batch(3)
        dp_numbers = self._get_datapipe(range=100)
        self.assertEqual(list(dp_numbers), list(df_numbers.unbatch(2)))

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_filter(self):
        if False:
            for i in range(10):
                print('nop')
        df_numbers = self._get_dataframes_pipe(range=10).filter(lambda x: x.i > 5)
        actual = list(df_numbers)
        self.assertEqual([(6, 0), (7, 1), (8, 2), (9, 0)], actual)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_collate(self):
        if False:
            for i in range(10):
                print('nop')

        def collate_i(column):
            if False:
                print('Hello World!')
            return column.sum()

        def collate_j(column):
            if False:
                i = 10
                return i + 15
            return column.prod()
        df_numbers = self._get_dataframes_pipe(range=30).batch(3)
        df_numbers = df_numbers.collate({'j': collate_j, 'i': collate_i})
        expected_i = [3, 12, 21, 30, 39, 48, 57, 66, 75, 84]
        actual_i = []
        for (i, j) in df_numbers:
            actual_i.append(i)
        self.assertEqual(expected_i, actual_i)
        actual_i = []
        for item in df_numbers:
            actual_i.append(item.i)
        self.assertEqual(expected_i, actual_i)

class IDP_NoLen(IterDataPipe):

    def __init__(self, input_dp):
        if False:
            while True:
                i = 10
        super().__init__()
        self.input_dp = input_dp

    def __iter__(self):
        if False:
            print('Hello World!')
        input_dp = self.input_dp if isinstance(self.input_dp, IterDataPipe) else copy.deepcopy(self.input_dp)
        yield from input_dp

def _fake_fn(data):
    if False:
        i = 10
        return i + 15
    return data

def _fake_add(constant, data):
    if False:
        print('Hello World!')
    return constant + data

def _fake_filter_fn(data):
    if False:
        for i in range(10):
            print('nop')
    return True

def _simple_filter_fn(data):
    if False:
        while True:
            i = 10
    return data >= 5

def _fake_filter_fn_constant(constant, data):
    if False:
        print('Hello World!')
    return data >= constant

def _mul_10(x):
    if False:
        print('Hello World!')
    return x * 10

def _mod_3_test(x):
    if False:
        return 10
    return x % 3 == 1

def _to_list(x):
    if False:
        return 10
    return [x]
lambda_fn1 = lambda x: x
lambda_fn2 = lambda x: x % 2
lambda_fn3 = lambda x: x >= 5

class Add1Module(nn.Module):

    def forward(self, x):
        if False:
            while True:
                i = 10
        return x + 1

class Add1Callable:

    def __call__(self, x):
        if False:
            print('Hello World!')
        return x + 1

class TestFunctionalIterDataPipe(TestCase):

    def _serialization_test_helper(self, datapipe, use_dill):
        if False:
            while True:
                i = 10
        if use_dill:
            serialized_dp = dill.dumps(datapipe)
            deserialized_dp = dill.loads(serialized_dp)
        else:
            serialized_dp = pickle.dumps(datapipe)
            deserialized_dp = pickle.loads(serialized_dp)
        try:
            self.assertEqual(list(datapipe), list(deserialized_dp))
        except AssertionError as e:
            print(f'{datapipe} is failing.')
            raise e

    def _serialization_test_for_single_dp(self, dp, use_dill=False):
        if False:
            print('Hello World!')
        self._serialization_test_helper(dp, use_dill)
        it = iter(dp)
        _ = next(it)
        self._serialization_test_helper(dp, use_dill)
        it = iter(dp)
        _ = list(it)
        self._serialization_test_helper(dp, use_dill)

    def _serialization_test_for_dp_with_children(self, dp1, dp2, use_dill=False):
        if False:
            while True:
                i = 10
        self._serialization_test_helper(dp1, use_dill)
        self._serialization_test_helper(dp2, use_dill)
        (it1, it2) = (iter(dp1), iter(dp2))
        (_, _) = (next(it1), next(it2))
        with warnings.catch_warnings(record=True) as wa:
            self._serialization_test_helper(dp1, use_dill)
            self._serialization_test_helper(dp2, use_dill)
        it1 = iter(dp1)
        _ = list(it1)
        with warnings.catch_warnings(record=True) as wa:
            self._serialization_test_helper(dp1, use_dill)
            self._serialization_test_helper(dp2, use_dill)
        it2 = iter(dp2)
        _ = list(it2)
        self._serialization_test_helper(dp1, use_dill)
        self._serialization_test_helper(dp2, use_dill)

    def test_serializable(self):
        if False:
            while True:
                i = 10
        picklable_datapipes: List = [(dp.iter.Batcher, None, (3, True), {}), (dp.iter.Collator, None, (_fake_fn,), {}), (dp.iter.Concater, None, (dp.iter.IterableWrapper(range(5)),), {}), (dp.iter.Demultiplexer, None, (2, _simple_filter_fn), {}), (dp.iter.FileLister, '.', (), {}), (dp.iter.FileOpener, None, (), {}), (dp.iter.Filter, None, (_fake_filter_fn,), {}), (dp.iter.Filter, None, (partial(_fake_filter_fn_constant, 5),), {}), (dp.iter.Forker, None, (2,), {}), (dp.iter.Forker, None, (2,), {'copy': 'shallow'}), (dp.iter.Grouper, None, (_fake_filter_fn,), {'group_size': 2}), (dp.iter.IterableWrapper, range(10), (), {}), (dp.iter.Mapper, None, (_fake_fn,), {}), (dp.iter.Mapper, None, (partial(_fake_add, 1),), {}), (dp.iter.Multiplexer, None, (dp.iter.IterableWrapper(range(10)),), {}), (dp.iter.Sampler, None, (), {}), (dp.iter.Shuffler, dp.iter.IterableWrapper([0] * 10), (), {}), (dp.iter.StreamReader, None, (), {}), (dp.iter.UnBatcher, None, (0,), {}), (dp.iter.Zipper, None, (dp.iter.IterableWrapper(range(10)),), {})]
        dp_skip_comparison = {dp.iter.FileOpener, dp.iter.StreamReader}
        dp_compare_children = {dp.iter.Demultiplexer, dp.iter.Forker}
        for (dpipe, custom_input, dp_args, dp_kwargs) in picklable_datapipes:
            if custom_input is None:
                custom_input = dp.iter.IterableWrapper(range(10))
            if dpipe in dp_skip_comparison:
                datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)
                serialized_dp = pickle.dumps(datapipe)
                _ = pickle.loads(serialized_dp)
            elif dpipe in dp_compare_children:
                (dp1, dp2) = dpipe(custom_input, *dp_args, **dp_kwargs)
                self._serialization_test_for_dp_with_children(dp1, dp2)
            else:
                datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)
                self._serialization_test_for_single_dp(datapipe)

    def test_serializable_with_dill(self):
        if False:
            print('Hello World!')
        'Only for DataPipes that take in a function as argument'
        input_dp = dp.iter.IterableWrapper(range(10))
        datapipes_with_lambda_fn: List[Tuple[Type[IterDataPipe], Tuple, Dict[str, Any]]] = [(dp.iter.Collator, (lambda_fn1,), {}), (dp.iter.Demultiplexer, (2, lambda_fn2), {}), (dp.iter.Filter, (lambda_fn3,), {}), (dp.iter.Grouper, (lambda_fn3,), {}), (dp.iter.Mapper, (lambda_fn1,), {})]

        def _local_fns():
            if False:
                print('Hello World!')

            def _fn1(x):
                if False:
                    return 10
                return x

            def _fn2(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x % 2

            def _fn3(x):
                if False:
                    return 10
                return x >= 5
            return (_fn1, _fn2, _fn3)
        (fn1, fn2, fn3) = _local_fns()
        datapipes_with_local_fn: List[Tuple[Type[IterDataPipe], Tuple, Dict[str, Any]]] = [(dp.iter.Collator, (fn1,), {}), (dp.iter.Demultiplexer, (2, fn2), {}), (dp.iter.Filter, (fn3,), {}), (dp.iter.Grouper, (fn3,), {}), (dp.iter.Mapper, (fn1,), {})]
        dp_compare_children = {dp.iter.Demultiplexer}
        if HAS_DILL:
            for (dpipe, dp_args, dp_kwargs) in datapipes_with_lambda_fn + datapipes_with_local_fn:
                if dpipe in dp_compare_children:
                    (dp1, dp2) = dpipe(input_dp, *dp_args, **dp_kwargs)
                    self._serialization_test_for_dp_with_children(dp1, dp2, use_dill=True)
                else:
                    datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)
                    self._serialization_test_for_single_dp(datapipe, use_dill=True)
        else:
            msgs = ('^Lambda function is not supported by pickle', '^Local function is not supported by pickle')
            for (dps, msg) in zip((datapipes_with_lambda_fn, datapipes_with_local_fn), msgs):
                for (dpipe, dp_args, dp_kwargs) in dps:
                    with self.assertWarnsRegex(UserWarning, msg):
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)
                    with self.assertRaises((pickle.PicklingError, AttributeError)):
                        pickle.dumps(datapipe)

    def test_docstring(self):
        if False:
            return 10
        '\n        Ensure functional form of IterDataPipe has the correct docstring from\n        the class form.\n\n        Regression test for https://github.com/pytorch/data/issues/792.\n        '
        input_dp = dp.iter.IterableWrapper(range(10))
        for dp_funcname in ['batch', 'collate', 'concat', 'demux', 'filter', 'fork', 'map', 'mux', 'read_from_stream', 'shuffle', 'unbatch', 'zip']:
            if sys.version_info >= (3, 9):
                docstring = pydoc.render_doc(thing=getattr(input_dp, dp_funcname), forceload=True)
            elif sys.version_info < (3, 9):
                docstring = getattr(input_dp, dp_funcname).__doc__
            assert f'(functional name: ``{dp_funcname}``)' in docstring
            assert 'Args:' in docstring
            assert 'Example:' in docstring or 'Examples:' in docstring

    def test_iterable_wrapper_datapipe(self):
        if False:
            print('Hello World!')
        input_ls = list(range(10))
        input_dp = dp.iter.IterableWrapper(input_ls)
        self.assertEqual(input_ls, list(input_dp))
        it = iter(input_dp)
        self.assertEqual(0, next(it))
        input_ls.append(50)
        self.assertEqual(list(range(1, 10)), list(it))
        input_ls2 = [1, 2, 3]
        input_dp_shallow = dp.iter.IterableWrapper(input_ls2, deepcopy=False)
        input_ls2.append(10)
        self.assertEqual([1, 2, 3, 10], list(input_dp_shallow))
        input_ls = list(range(10))
        input_dp = dp.iter.IterableWrapper(input_ls)
        n_elements_before_reset = 5
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(input_dp, n_elements_before_reset)
        self.assertEqual(input_ls[:n_elements_before_reset], res_before_reset)
        self.assertEqual(input_ls, res_after_reset)
        self.assertEqual(len(input_ls), len(input_dp))

    def test_concat_iterdatapipe(self):
        if False:
            for i in range(10):
                print('nop')
        input_dp1 = dp.iter.IterableWrapper(range(10))
        input_dp2 = dp.iter.IterableWrapper(range(5))
        with self.assertRaisesRegex(ValueError, 'Expected at least one DataPipe'):
            dp.iter.Concater()
        with self.assertRaisesRegex(TypeError, 'Expected all inputs to be `IterDataPipe`'):
            dp.iter.Concater(input_dp1, ())
        concat_dp = input_dp1.concat(input_dp2)
        self.assertEqual(len(concat_dp), 15)
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))
        n_elements_before_reset = 5
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(concat_dp, n_elements_before_reset)
        self.assertEqual(list(range(5)), res_before_reset)
        self.assertEqual(list(range(10)) + list(range(5)), res_after_reset)
        input_dp_nl = IDP_NoLen(range(5))
        concat_dp = input_dp1.concat(input_dp_nl)
        with self.assertRaisesRegex(TypeError, "instance doesn't have valid length$"):
            len(concat_dp)
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

    def test_fork_iterdatapipe(self):
        if False:
            for i in range(10):
                print('nop')
        input_dp = dp.iter.IterableWrapper(range(10))
        with self.assertRaises(ValueError):
            input_dp.fork(num_instances=0)
        dp0 = input_dp.fork(num_instances=1, buffer_size=0)
        self.assertEqual(dp0, input_dp)
        (dp1, dp2, dp3) = input_dp.fork(num_instances=3)
        self.assertTrue(all((n1 is n2 and n1 is n3 for (n1, n2, n3) in zip(dp1, dp2, dp3))))
        (output1, output2, output3) = (list(dp1), list(dp2), list(dp3))
        self.assertEqual(list(range(10)), output1)
        self.assertEqual(list(range(10)), output2)
        self.assertEqual(list(range(10)), output3)
        (dp1, dp2) = input_dp.fork(num_instances=2)
        output = []
        for (n1, n2) in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i) for i in range(10)], output)
        (dp1, dp2) = input_dp.fork(num_instances=2, buffer_size=4)
        it1 = iter(dp1)
        for _ in range(4):
            next(it1)
        with self.assertRaises(BufferError):
            next(it1)
        with self.assertRaises(BufferError):
            list(dp2)
        (dp1, dp2) = input_dp.fork(num_instances=2, buffer_size=5)
        with self.assertRaises(BufferError):
            list(dp2)
        with warnings.catch_warnings(record=True) as wa:
            (dp1, dp2) = input_dp.fork(num_instances=2, buffer_size=-1)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), 'Unlimited buffer size is set')
        (l1, l2) = (list(dp1), list(dp2))
        for (d1, d2) in zip(l1, l2):
            self.assertEqual(d1, d2)
        (dp1, dp2) = input_dp.fork(num_instances=2, buffer_size=1)
        output = []
        for (n1, n2) in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i) for i in range(10)], output)
        (dp1, dp2) = input_dp.map(_to_list).fork(num_instances=2, copy='shallow')
        for (n1, n2) in zip(dp1, dp2):
            self.assertIsNot(n1, n2)
            self.assertEqual(n1, n2)
        (dp1, dp2) = input_dp.map(_to_list).map(_to_list).fork(num_instances=2, copy='deep')
        for (n1, n2) in zip(dp1, dp2):
            self.assertIsNot(n1[0], n2[0])
            self.assertEqual(n1, n2)
        with self.assertRaises(ValueError):
            input_dp.fork(num_instances=2, copy='unknown')
        (dp1, dp2, dp3) = input_dp.fork(num_instances=3)
        (output1, output2, output3) = ([], [], [])
        for (i, (n1, n2)) in enumerate(zip(dp1, dp2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:
                output3 = list(dp3)
                break
        self.assertEqual(list(range(5)), output1)
        self.assertEqual(list(range(5)), output2)
        self.assertEqual(list(range(10)), output3)
        (dp1, dp2) = input_dp.fork(num_instances=2)
        _ = iter(dp1)
        output2 = []
        with self.assertRaisesRegex(RuntimeError, 'iterator has been invalidated'):
            for (i, n2) in enumerate(dp2):
                output2.append(n2)
                if i == 4:
                    with warnings.catch_warnings(record=True) as wa:
                        _ = iter(dp1)
                        self.assertEqual(len(wa), 1)
                        self.assertRegex(str(wa[0].message), 'child DataPipes are not exhausted')
        self.assertEqual(list(range(5)), output2)
        (dp1, dp2) = input_dp.fork(num_instances=2)
        (output1, output2) = ([], [])
        for (i, (n1, n2)) in enumerate(zip(dp1, dp2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    _ = iter(dp1)
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), 'Some child DataPipes are not exhausted')
                break
        with warnings.catch_warnings(record=True) as wa:
            for (i, (n1, n2)) in enumerate(zip(dp1, dp2)):
                output1.append(n1)
                output2.append(n2)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), 'child DataPipes are not exhausted')
        self.assertEqual(list(range(5)) + list(range(10)), output1)
        self.assertEqual(list(range(5)) + list(range(10)), output2)
        (dp1, dp2, dp3) = input_dp.fork(num_instances=3)
        (output1, output2) = (list(dp1), list(dp2))
        self.assertEqual(list(range(10)), output1)
        self.assertEqual(list(range(10)), output2)
        with warnings.catch_warnings(record=True) as wa:
            self.assertEqual(list(range(10)), list(dp1))
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), 'Some child DataPipes are not exhausted')
        output3 = []
        for (i, n3) in enumerate(dp3):
            output3.append(n3)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    output1 = list(dp1)
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), 'Some child DataPipes are not exhausted')
                self.assertEqual(list(range(5)), output3)
                self.assertEqual(list(range(10)), output1)
                break
        self.assertEqual(list(range(10)), list(dp3))
        (dp1, dp2, dp3) = input_dp.fork(num_instances=3)
        self.assertEqual(len(input_dp), len(dp1))
        self.assertEqual(len(input_dp), len(dp2))
        self.assertEqual(len(input_dp), len(dp3))
        (dp1, dp2, dp3) = input_dp.fork(num_instances=3)
        traverse_dps(dp1)
        for _ in zip(dp1, dp2, dp3):
            pass
        traverse_dps(dp2)

    def test_mux_iterdatapipe(self):
        if False:
            return 10
        input_dp1 = dp.iter.IterableWrapper(range(4))
        input_dp2 = dp.iter.IterableWrapper(range(4, 8))
        input_dp3 = dp.iter.IterableWrapper(range(8, 12))
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        expected_output = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))
        input_dp1 = dp.iter.IterableWrapper([1, 2, 3, 4])
        input_dp2 = dp.iter.IterableWrapper([10])
        input_dp3 = dp.iter.IterableWrapper([100, 200, 300])
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        expected_output = [1, 10, 100]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))
        input_dp1 = dp.iter.IterableWrapper([0, 1, 2, 3])
        input_dp2 = dp.iter.IterableWrapper([])
        output_dp = input_dp1.mux(input_dp2)
        self.assertEqual(len(input_dp2), len(output_dp))
        self.assertEqual(list(input_dp2), list(output_dp))
        input_dp1 = dp.iter.IterableWrapper(range(10))
        input_dp_no_len = IDP_NoLen(range(10))
        output_dp = input_dp1.mux(input_dp_no_len)
        with self.assertRaises(TypeError):
            len(output_dp)

    def test_demux_iterdatapipe(self):
        if False:
            i = 10
            return i + 15
        input_dp = dp.iter.IterableWrapper(range(10))
        with self.assertRaises(ValueError):
            input_dp.demux(num_instances=0, classifier_fn=lambda x: 0)
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        (output1, output2) = (list(dp1), list(dp2))
        self.assertEqual(list(range(0, 10, 2)), output1)
        self.assertEqual(list(range(1, 10, 2)), output2)
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output = []
        for (n1, n2) in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i + 1) for i in range(0, 10, 2)], output)
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: 0 if x >= 5 else 1, buffer_size=4)
        it1 = iter(dp1)
        with self.assertRaises(BufferError):
            next(it1)
        with self.assertRaises(BufferError):
            list(dp2)
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: 0 if x >= 5 else 1, buffer_size=5)
        (output1, output2) = (list(dp1), list(dp2))
        self.assertEqual(list(range(5, 10)), output1)
        self.assertEqual(list(range(0, 5)), output2)
        with warnings.catch_warnings(record=True) as wa:
            (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: 0 if x >= 5 else 1, buffer_size=-1)
            exp_l = 1 if HAS_DILL else 2
            self.assertEqual(len(wa), exp_l)
            self.assertRegex(str(wa[-1].message), 'Unlimited buffer size is set')
        (output1, output2) = (list(dp1), list(dp2))
        self.assertEqual(list(range(5, 10)), output1)
        self.assertEqual(list(range(0, 5)), output2)
        dp0 = input_dp.demux(num_instances=1, classifier_fn=lambda x: x % 2)
        it = iter(dp0[0])
        with self.assertRaises(ValueError):
            next(it)
            next(it)
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        _ = iter(dp1)
        output2 = []
        with self.assertRaisesRegex(RuntimeError, 'iterator has been invalidated'):
            for (i, n2) in enumerate(dp2):
                output2.append(n2)
                if i == 4:
                    with warnings.catch_warnings(record=True) as wa:
                        _ = iter(dp1)
                        self.assertEqual(len(wa), 1)
                        self.assertRegex(str(wa[0].message), 'child DataPipes are not exhausted')
        self.assertEqual(list(range(1, 10, 2)), output2)
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        (output1, output2) = ([], [])
        for (n1, n2) in zip(dp1, dp2):
            output1.append(n1)
            output2.append(n2)
            if n1 == 4:
                break
        with warnings.catch_warnings(record=True) as wa:
            i1 = iter(dp1)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), 'Some child DataPipes are not exhausted')
            for (n1, n2) in zip(dp1, dp2):
                output1.append(n1)
                output2.append(n2)
            self.assertEqual([0, 2, 4] + list(range(0, 10, 2)), output1)
            self.assertEqual([1, 3, 5] + list(range(1, 10, 2)), output2)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), 'child DataPipes are not exhausted')
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output1 = list(dp1)
        self.assertEqual(list(range(0, 10, 2)), output1)
        with warnings.catch_warnings(record=True) as wa:
            self.assertEqual(list(range(0, 10, 2)), list(dp1))
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), 'Some child DataPipes are not exhausted')
        output2 = []
        for (i, n2) in enumerate(dp2):
            output2.append(n2)
            if i == 1:
                self.assertEqual(list(range(1, 5, 2)), output2)
                with warnings.catch_warnings(record=True) as wa:
                    self.assertEqual(list(range(0, 10, 2)), list(dp1))
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), 'Some child DataPipes are not exhausted')
                break
        output2 = list(dp2)
        self.assertEqual(list(range(1, 10, 2)), output2)
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2 if x % 5 != 0 else None, drop_none=True)
        self.assertEqual([2, 4, 6, 8], list(dp1))
        self.assertEqual([1, 3, 7, 9], list(dp2))
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2 if x % 5 != 0 else None, drop_none=False)
        it1 = iter(dp1)
        with self.assertRaises(ValueError):
            next(it1)
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        with self.assertRaises(TypeError):
            len(dp1)
        with self.assertRaises(TypeError):
            len(dp2)
        (dp1, dp2) = input_dp.demux(num_instances=2, classifier_fn=odd_or_even)
        traverse_dps(dp1)
        for _ in zip(dp1, dp2):
            pass
        traverse_dps(dp2)

    def test_map_iterdatapipe(self):
        if False:
            return 10
        target_length = 10
        input_dp = dp.iter.IterableWrapper(range(target_length))

        def fn(item, dtype=torch.float, *, sum=False):
            if False:
                while True:
                    i = 10
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()
        map_dp = input_dp.map(fn)
        self.assertEqual(target_length, len(map_dp))
        for (x, y) in zip(map_dp, range(target_length)):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))
        map_dp = input_dp.map(partial(fn, dtype=torch.int, sum=True))
        for (x, y) in zip(map_dp, range(target_length)):
            self.assertEqual(x, torch.tensor(y, dtype=torch.int).sum())
        self.assertEqual(target_length, len(map_dp))
        input_dp_nl = IDP_NoLen(range(target_length))
        map_dp_nl = input_dp_nl.map(lambda x: x)
        for (x, y) in zip(map_dp_nl, range(target_length)):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))
        with self.assertRaisesRegex(TypeError, "instance doesn't have valid length$"):
            len(map_dp_nl)
        n_elements_before_reset = 5
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(map_dp, n_elements_before_reset)
        self.assertEqual(list(range(n_elements_before_reset)), res_before_reset)
        self.assertEqual(list(range(10)), res_after_reset)

    @suppress_warnings
    def test_map_tuple_list_with_col_iterdatapipe(self):
        if False:
            i = 10
            return i + 15

        def fn_11(d):
            if False:
                for i in range(10):
                    print('nop')
            return -d

        def fn_1n(d):
            if False:
                for i in range(10):
                    print('nop')
            return (-d, d)

        def fn_n1(d0, d1):
            if False:
                for i in range(10):
                    print('nop')
            return d0 + d1

        def fn_nn(d0, d1):
            if False:
                for i in range(10):
                    print('nop')
            return (-d0, -d1, d0 + d1)

        def fn_n1_def(d0, d1=1):
            if False:
                print('Hello World!')
            return d0 + d1

        def fn_n1_kwargs(d0, d1, **kwargs):
            if False:
                print('Hello World!')
            return d0 + d1

        def fn_n1_pos(d0, d1, *args):
            if False:
                while True:
                    i = 10
            return d0 + d1

        def fn_n1_sep_pos(d0, *args, d1):
            if False:
                i = 10
                return i + 15
            return d0 + d1

        def fn_cmplx(d0, d1=1, *args, d2, **kwargs):
            if False:
                while True:
                    i = 10
            return d0 + d1
        p_fn_n1 = partial(fn_n1, d1=1)
        p_fn_cmplx = partial(fn_cmplx, d2=2)
        p_fn_cmplx_large_arg = partial(fn_cmplx, d2={i: list(range(i)) for i in range(10000)})

        def _helper(ref_fn, fn, input_col=None, output_col=None, error=None):
            if False:
                return 10
            for constr in (list, tuple):
                datapipe = dp.iter.IterableWrapper([constr((0, 1, 2)), constr((3, 4, 5)), constr((6, 7, 8))])
                if ref_fn is None:
                    with self.assertRaises(error):
                        res_dp = datapipe.map(fn, input_col, output_col)
                        list(res_dp)
                else:
                    res_dp = datapipe.map(fn, input_col, output_col)
                    ref_dp = datapipe.map(ref_fn)
                    self.assertEqual(list(res_dp), list(ref_dp))
                    self.assertEqual(list(res_dp), list(ref_dp))
        _helper(lambda data: data, fn_n1_def, 0, 1)
        _helper(lambda data: (data[0], data[1], data[0] + data[1]), fn_n1_def, [0, 1], 2)
        _helper(lambda data: data, p_fn_n1, 0, 1)
        _helper(lambda data: data, p_fn_cmplx, 0, 1)
        _helper(lambda data: data, p_fn_cmplx_large_arg, 0, 1)
        _helper(lambda data: (data[0], data[1], data[0] + data[1]), p_fn_cmplx, [0, 1], 2)
        _helper(lambda data: (data[0] + data[1],), fn_n1_pos, [0, 1, 2])
        _helper(lambda data: (data[0], -data[1], data[2]), fn_11, 1)
        _helper(lambda data: (data[0], (-data[1], data[1]), data[2]), fn_1n, 1)
        _helper(None, fn_1n, 3, error=IndexError)
        _helper(None, fn_n1, 1, error=ValueError)
        _helper(None, fn_n1, [0, 1, 2], error=ValueError)
        _helper(None, lambda d0, d1: d0 + d1, 0, error=ValueError)
        _helper(None, lambda d0, d1: d0 + d1, [0, 1, 2], error=ValueError)
        _helper(None, fn_cmplx, 0, 1, ValueError)
        _helper(None, fn_n1_pos, 1, error=ValueError)
        _helper(None, fn_n1_def, [0, 1, 2], 1, error=ValueError)
        _helper(None, p_fn_n1, [0, 1], error=ValueError)
        _helper(None, fn_1n, [1, 2], error=ValueError)
        _helper(None, fn_n1_sep_pos, [0, 1, 2], error=ValueError)
        _helper(None, fn_n1_kwargs, 1, error=ValueError)
        _helper(None, fn_cmplx, [0, 1], 2, ValueError)
        _helper(lambda data: (data[1], data[2] + data[0]), fn_n1, [2, 0])
        _helper(lambda data: (data[0], (-data[2], -data[1], data[2] + data[1])), fn_nn, [2, 1])
        _helper(None, fn_n1, None, 1, error=ValueError)
        _helper(None, fn_n1, None, [0, 1], error=ValueError)
        _helper(lambda data: (-data[1], data[1], data[2]), fn_11, 1, [0])
        _helper(lambda data: (-data[1], data[1], data[2]), fn_11, 1, 0)
        _helper(lambda data: (data[0], data[1], (-data[1], data[1])), fn_1n, 1, 2)
        _helper(None, fn_1n, 1, 3, error=IndexError)
        _helper(lambda data: (data[0], data[0] + data[2], data[2]), fn_n1, [0, 2], 1)
        _helper(lambda data: ((-data[1], -data[2], data[1] + data[2]), data[1], data[2]), fn_nn, [1, 2], 0)
        _helper(lambda data: (*data, -data[1]), fn_11, 1, -1)
        _helper(lambda data: (*data, (-data[1], data[1])), fn_1n, 1, -1)
        _helper(lambda data: (*data, data[0] + data[2]), fn_n1, [0, 2], -1)
        _helper(lambda data: (*data, (-data[1], -data[2], data[1] + data[2])), fn_nn, [1, 2], -1)
        _helper(lambda data: (str(data[0]), data[1], data[2]), str, 0)
        _helper(lambda data: (data[0], data[1], int(data[2])), int, 2)
        _helper(lambda data: (data[0] + 1, data[1], data[2]), Add1Module(), 0)
        _helper(lambda data: (data[0] + 1, data[1], data[2]), Add1Callable(), 0)

    @suppress_warnings
    @skipIfTorchDynamo
    def test_map_dict_with_col_iterdatapipe(self):
        if False:
            i = 10
            return i + 15

        def fn_11(d):
            if False:
                print('Hello World!')
            return -d

        def fn_1n(d):
            if False:
                return 10
            return (-d, d)

        def fn_n1(d0, d1):
            if False:
                while True:
                    i = 10
            return d0 + d1

        def fn_nn(d0, d1):
            if False:
                print('Hello World!')
            return (-d0, -d1, d0 + d1)

        def fn_n1_def(d0, d1=1):
            if False:
                for i in range(10):
                    print('nop')
            return d0 + d1
        p_fn_n1 = partial(fn_n1, d1=1)

        def fn_n1_pos(d0, d1, *args):
            if False:
                print('Hello World!')
            return d0 + d1

        def fn_n1_kwargs(d0, d1, **kwargs):
            if False:
                while True:
                    i = 10
            return d0 + d1

        def fn_kwonly(*, d0, d1):
            if False:
                print('Hello World!')
            return d0 + d1

        def fn_has_nondefault_kwonly(d0, *, d1):
            if False:
                while True:
                    i = 10
            return d0 + d1

        def fn_cmplx(d0, d1=1, *args, d2, **kwargs):
            if False:
                while True:
                    i = 10
            return d0 + d1
        p_fn_cmplx = partial(fn_cmplx, d2=2)
        p_fn_cmplx_large_arg = partial(fn_cmplx, d2={i: list(range(i)) for i in range(10000)})

        def _dict_update(data, newdata, remove_idx=None):
            if False:
                while True:
                    i = 10
            _data = dict(data)
            _data.update(newdata)
            if remove_idx:
                for idx in remove_idx:
                    del _data[idx]
            return _data

        def _helper(ref_fn, fn, input_col=None, output_col=None, error=None):
            if False:
                return 10
            datapipe = dp.iter.IterableWrapper([{'x': 0, 'y': 1, 'z': 2}, {'x': 3, 'y': 4, 'z': 5}, {'x': 6, 'y': 7, 'z': 8}])
            if ref_fn is None:
                with self.assertRaises(error):
                    res_dp = datapipe.map(fn, input_col, output_col)
                    list(res_dp)
            else:
                res_dp = datapipe.map(fn, input_col, output_col)
                ref_dp = datapipe.map(ref_fn)
                self.assertEqual(list(res_dp), list(ref_dp))
                self.assertEqual(list(res_dp), list(ref_dp))
        _helper(lambda data: data, fn_n1_def, 'x', 'y')
        _helper(lambda data: data, p_fn_n1, 'x', 'y')
        _helper(lambda data: data, p_fn_cmplx, 'x', 'y')
        _helper(lambda data: data, p_fn_cmplx_large_arg, 'x', 'y')
        _helper(lambda data: _dict_update(data, {'z': data['x'] + data['y']}), p_fn_cmplx, ['x', 'y', 'z'], 'z')
        _helper(lambda data: _dict_update(data, {'z': data['x'] + data['y']}), fn_n1_def, ['x', 'y'], 'z')
        _helper(None, fn_n1_pos, 'x', error=ValueError)
        _helper(None, fn_n1_kwargs, 'x', error=ValueError)
        _helper(None, fn_kwonly, ['x', 'y'], error=ValueError)
        _helper(None, fn_has_nondefault_kwonly, ['x', 'y'], error=ValueError)
        _helper(None, fn_cmplx, ['x', 'y'], error=ValueError)
        _helper(lambda data: _dict_update(data, {'y': -data['y']}), fn_11, 'y')
        _helper(lambda data: _dict_update(data, {'y': (-data['y'], data['y'])}), fn_1n, 'y')
        _helper(None, fn_1n, 'a', error=KeyError)
        _helper(None, fn_n1, 'y', error=ValueError)
        _helper(None, fn_1n, ['x', 'y'], error=ValueError)
        _helper(None, fn_n1_def, ['x', 'y', 'z'], error=ValueError)
        _helper(None, p_fn_n1, ['x', 'y'], error=ValueError)
        _helper(None, fn_n1_kwargs, ['x', 'y', 'z'], error=ValueError)
        _helper(lambda data: _dict_update(data, {'z': data['x'] + data['z']}, ['x']), fn_n1, ['z', 'x'])
        _helper(lambda data: _dict_update(data, {'z': (-data['z'], -data['y'], data['y'] + data['z'])}, ['y']), fn_nn, ['z', 'y'])
        _helper(None, fn_n1, None, 'x', error=ValueError)
        _helper(None, fn_n1, None, ['x', 'y'], error=ValueError)
        _helper(lambda data: _dict_update(data, {'x': -data['y']}), fn_11, 'y', ['x'])
        _helper(lambda data: _dict_update(data, {'x': -data['y']}), fn_11, 'y', 'x')
        _helper(lambda data: _dict_update(data, {'z': (-data['y'], data['y'])}), fn_1n, 'y', 'z')
        _helper(lambda data: _dict_update(data, {'y': data['x'] + data['z']}), fn_n1, ['x', 'z'], 'y')
        _helper(lambda data: _dict_update(data, {'x': (-data['y'], -data['z'], data['y'] + data['z'])}), fn_nn, ['y', 'z'], 'x')
        _helper(lambda data: _dict_update(data, {'a': -data['y']}), fn_11, 'y', 'a')
        _helper(lambda data: _dict_update(data, {'a': (-data['y'], data['y'])}), fn_1n, 'y', 'a')
        _helper(lambda data: _dict_update(data, {'a': data['x'] + data['z']}), fn_n1, ['x', 'z'], 'a')
        _helper(lambda data: _dict_update(data, {'a': (-data['y'], -data['z'], data['y'] + data['z'])}), fn_nn, ['y', 'z'], 'a')

    def test_collate_iterdatapipe(self):
        if False:
            while True:
                i = 10
        arrs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        input_dp = dp.iter.IterableWrapper(arrs)

        def _collate_fn(batch, default_type=torch.float):
            if False:
                while True:
                    i = 10
            return torch.tensor(sum(batch), dtype=default_type)
        collate_dp = input_dp.collate()
        for (x, y) in zip(arrs, collate_dp):
            self.assertEqual(torch.tensor(x), y)
        collate_dp = input_dp.collate(collate_fn=_collate_fn)
        for (x, y) in zip(arrs, collate_dp):
            self.assertEqual(torch.tensor(sum(x), dtype=torch.float), y)
        collate_dp = input_dp.collate(partial(_collate_fn, default_type=torch.int))
        for (x, y) in zip(arrs, collate_dp):
            self.assertEqual(torch.tensor(sum(x), dtype=torch.int), y)
        n_elements_before_reset = 1
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(collate_dp, n_elements_before_reset)
        self.assertEqual([torch.tensor(6, dtype=torch.int)], res_before_reset)
        for (x, y) in zip(arrs, res_after_reset):
            self.assertEqual(torch.tensor(sum(x), dtype=torch.int), y)
        self.assertEqual(len(input_dp), len(collate_dp))
        input_dp_nl = IDP_NoLen(arrs)
        collate_dp_nl = input_dp_nl.collate()
        with self.assertRaisesRegex(TypeError, "instance doesn't have valid length$"):
            len(collate_dp_nl)
        for (x, y) in zip(arrs, collate_dp_nl):
            self.assertEqual(torch.tensor(x), y)

    def test_batch_iterdatapipe(self):
        if False:
            return 10
        arrs = list(range(10))
        input_dp = dp.iter.IterableWrapper(arrs)
        with self.assertRaises(AssertionError):
            input_dp.batch(batch_size=0)
        bs = 3
        batch_dp = input_dp.batch(batch_size=bs)
        self.assertEqual(len(batch_dp), 4)
        for (i, batch) in enumerate(batch_dp):
            self.assertEqual(len(batch), 1 if i == 3 else bs)
            self.assertEqual(batch, arrs[i * bs:i * bs + len(batch)])
        bs = 4
        batch_dp = input_dp.batch(batch_size=bs, drop_last=True)
        for (i, batch) in enumerate(batch_dp):
            self.assertEqual(batch, arrs[i * bs:i * bs + len(batch)])
        for (i, batch) in enumerate(batch_dp):
            self.assertEqual(len(batch), bs)
        self.assertEqual(len(batch_dp), 2)
        input_dp_nl = IDP_NoLen(range(10))
        batch_dp_nl = input_dp_nl.batch(batch_size=2)
        with self.assertRaisesRegex(TypeError, "instance doesn't have valid length$"):
            len(batch_dp_nl)
        n_elements_before_reset = 1
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(batch_dp, n_elements_before_reset)
        self.assertEqual([[0, 1, 2, 3]], res_before_reset)
        self.assertEqual([[0, 1, 2, 3], [4, 5, 6, 7]], res_after_reset)

    def test_unbatch_iterdatapipe(self):
        if False:
            i = 10
            return i + 15
        target_length = 6
        prebatch_dp = dp.iter.IterableWrapper(range(target_length))
        input_dp = prebatch_dp.batch(3)
        unbatch_dp = input_dp.unbatch()
        self.assertEqual(len(list(unbatch_dp)), target_length)
        for (i, res) in zip(range(target_length), unbatch_dp):
            self.assertEqual(i, res)
        input_dp = dp.iter.IterableWrapper([[0, 1, 2], [3, 4, 5]])
        unbatch_dp = input_dp.unbatch()
        self.assertEqual(len(list(unbatch_dp)), target_length)
        for (i, res) in zip(range(target_length), unbatch_dp):
            self.assertEqual(i, res)
        input_dp = dp.iter.IterableWrapper([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        unbatch_dp = input_dp.unbatch()
        expected_dp = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.assertEqual(len(list(unbatch_dp)), 4)
        for (j, res) in zip(expected_dp, unbatch_dp):
            self.assertEqual(j, res)
        unbatch_dp = input_dp.unbatch(unbatch_level=2)
        expected_dp2 = [0, 1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(len(list(unbatch_dp)), 8)
        for (i, res) in zip(expected_dp2, unbatch_dp):
            self.assertEqual(i, res)
        unbatch_dp = input_dp.unbatch(unbatch_level=-1)
        self.assertEqual(len(list(unbatch_dp)), 8)
        for (i, res) in zip(expected_dp2, unbatch_dp):
            self.assertEqual(i, res)
        input_dp = dp.iter.IterableWrapper([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            unbatch_dp = input_dp.unbatch(unbatch_level=-2)
            for i in unbatch_dp:
                print(i)
        with self.assertRaises(IndexError):
            unbatch_dp = input_dp.unbatch(unbatch_level=5)
            for i in unbatch_dp:
                print(i)
        input_dp = dp.iter.IterableWrapper([[0, 1, 2], [3, 4, 5]])
        unbatch_dp = input_dp.unbatch(unbatch_level=-1)
        n_elements_before_reset = 3
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(unbatch_dp, n_elements_before_reset)
        self.assertEqual([0, 1, 2], res_before_reset)
        self.assertEqual([0, 1, 2, 3, 4, 5], res_after_reset)

    def test_filter_datapipe(self):
        if False:
            return 10
        input_ds = dp.iter.IterableWrapper(range(10))

        def _filter_fn(data, val):
            if False:
                print('Hello World!')
            return data >= val
        filter_dp = input_ds.filter(partial(_filter_fn, val=5))
        self.assertEqual(list(filter_dp), list(range(5, 10)))

        def _non_bool_fn(data):
            if False:
                i = 10
                return i + 15
            return 1
        filter_dp = input_ds.filter(filter_fn=_non_bool_fn)
        with self.assertRaises(ValueError):
            temp = list(filter_dp)
        tuple_input_ds = dp.iter.IterableWrapper([(d - 1, d, d + 1) for d in range(10)])
        input_col_1_dp = tuple_input_ds.filter(partial(_filter_fn, val=5), input_col=1)
        self.assertEqual(list(input_col_1_dp), [(d - 1, d, d + 1) for d in range(5, 10)])

        def _mul_filter_fn(a, b):
            if False:
                i = 10
                return i + 15
            return a + b < 10
        input_col_2_dp = tuple_input_ds.filter(_mul_filter_fn, input_col=[0, 2])
        self.assertEqual(list(input_col_2_dp), [(d - 1, d, d + 1) for d in range(5)])
        with self.assertRaises(ValueError):
            tuple_input_ds.filter(_mul_filter_fn, input_col=0)
        p_mul_filter_fn = partial(_mul_filter_fn, b=1)
        out = tuple_input_ds.filter(p_mul_filter_fn, input_col=0)
        self.assertEqual(list(out), [(d - 1, d, d + 1) for d in range(10)])

        def _mul_filter_fn_with_defaults(a, b=1):
            if False:
                while True:
                    i = 10
            return a + b < 10
        out = tuple_input_ds.filter(_mul_filter_fn_with_defaults, input_col=0)
        self.assertEqual(list(out), [(d - 1, d, d + 1) for d in range(10)])

        def _mul_filter_fn_with_kw_only(*, a, b):
            if False:
                i = 10
                return i + 15
            return a + b < 10
        with self.assertRaises(ValueError):
            tuple_input_ds.filter(_mul_filter_fn_with_kw_only, input_col=0)

        def _mul_filter_fn_with_kw_only_1_default(*, a, b=1):
            if False:
                for i in range(10):
                    print('nop')
            return a + b < 10
        with self.assertRaises(ValueError):
            tuple_input_ds.filter(_mul_filter_fn_with_kw_only_1_default, input_col=0)
        with self.assertRaisesRegex(TypeError, 'has no len'):
            len(filter_dp)
        filter_dp = input_ds.filter(partial(_filter_fn, val=5))
        n_elements_before_reset = 3
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(filter_dp, n_elements_before_reset)
        self.assertEqual(list(range(5, 10))[:n_elements_before_reset], res_before_reset)
        self.assertEqual(list(range(5, 10)), res_after_reset)

    def test_sampler_iterdatapipe(self):
        if False:
            i = 10
            return i + 15
        input_dp = dp.iter.IterableWrapper(range(10))
        sampled_dp = dp.iter.Sampler(input_dp)
        self.assertEqual(len(sampled_dp), 10)
        for (i, x) in enumerate(sampled_dp):
            self.assertEqual(x, i)
        random_sampled_dp = dp.iter.Sampler(input_dp, sampler=RandomSampler, sampler_kwargs={'replacement': True})
        input_dp_nolen = IDP_NoLen(range(10))
        with self.assertRaises(AssertionError):
            sampled_dp = dp.iter.Sampler(input_dp_nolen)

    def test_stream_reader_iterdatapipe(self):
        if False:
            for i in range(10):
                print('nop')
        from io import StringIO
        input_dp = dp.iter.IterableWrapper([('f1', StringIO('abcde')), ('f2', StringIO('bcdef'))])
        expected_res = ['abcde', 'bcdef']
        dp1 = input_dp.read_from_stream()
        self.assertEqual([d[1] for d in dp1], expected_res)
        dp2 = input_dp.read_from_stream(chunk=1)
        self.assertEqual([d[1] for d in dp2], [c for s in expected_res for c in s])
        with self.assertRaises(TypeError):
            len(dp1)

    def test_shuffler_iterdatapipe(self):
        if False:
            while True:
                i = 10
        input_dp = dp.iter.IterableWrapper(list(range(10)))
        with self.assertRaises(AssertionError):
            shuffle_dp = input_dp.shuffle(buffer_size=0)
        shuffler_dp = input_dp.shuffle()
        self.assertEqual(set(range(10)), set(shuffler_dp))
        torch.manual_seed(123)
        shuffler_dp = input_dp.shuffle()
        res = list(shuffler_dp)
        torch.manual_seed(123)
        self.assertEqual(list(shuffler_dp), res)
        shuffler_dp = input_dp.shuffle().set_seed(123)
        res = list(shuffler_dp)
        shuffler_dp.set_seed(123)
        self.assertEqual(list(shuffler_dp), res)
        unshuffled_dp = input_dp.shuffle().set_shuffle(False)
        self.assertEqual(list(unshuffled_dp), list(input_dp))
        shuffler_dp = input_dp.shuffle()
        n_elements_before_reset = 5
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(shuffler_dp, n_elements_before_reset)
        self.assertEqual(5, len(res_before_reset))
        for x in res_before_reset:
            self.assertTrue(x in set(range(10)))
        self.assertEqual(set(range(10)), set(res_after_reset))
        shuffler_dp = input_dp.shuffle()
        self.assertEqual(10, len(shuffler_dp))
        exp = list(range(100))
        from torch.utils.data.datapipes._hook_iterator import _SnapshotState

        def _serialization_helper(bs):
            if False:
                while True:
                    i = 10
            shuffler_dp = input_dp.shuffle(buffer_size=bs)
            it = iter(shuffler_dp)
            for _ in range(2):
                next(it)
            shuffler_dp_copy = pickle.loads(pickle.dumps(shuffler_dp))
            _simple_graph_snapshot_restoration(shuffler_dp_copy.datapipe, shuffler_dp.datapipe._number_of_samples_yielded)
            exp = list(it)
            shuffler_dp_copy._snapshot_state = _SnapshotState.Restored
            self.assertEqual(exp, list(shuffler_dp_copy))
        buffer_sizes = [2, 5, 15]
        for bs in buffer_sizes:
            _serialization_helper(bs)

    def test_zip_iterdatapipe(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            dp.iter.Zipper(dp.iter.IterableWrapper(range(10)), list(range(10)))
        zipped_dp = dp.iter.Zipper(dp.iter.IterableWrapper(range(10)), IDP_NoLen(range(5)))
        with self.assertRaisesRegex(TypeError, "instance doesn't have valid length$"):
            len(zipped_dp)
        exp = [(i, i) for i in range(5)]
        self.assertEqual(list(zipped_dp), exp)
        zipped_dp = dp.iter.Zipper(dp.iter.IterableWrapper(range(10)), dp.iter.IterableWrapper(range(5)))
        self.assertEqual(len(zipped_dp), 5)
        n_elements_before_reset = 3
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(zipped_dp, n_elements_before_reset)
        expected_res = [(i, i) for i in range(5)]
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)

class TestFunctionalMapDataPipe(TestCase):

    def _serialization_test_helper(self, datapipe, use_dill):
        if False:
            for i in range(10):
                print('nop')
        if use_dill:
            serialized_dp = dill.dumps(datapipe)
            deserialized_dp = dill.loads(serialized_dp)
        else:
            serialized_dp = pickle.dumps(datapipe)
            deserialized_dp = pickle.loads(serialized_dp)
        try:
            self.assertEqual(list(datapipe), list(deserialized_dp))
        except AssertionError as e:
            print(f'{datapipe} is failing.')
            raise e

    def _serialization_test_for_single_dp(self, dp, use_dill=False):
        if False:
            while True:
                i = 10
        self._serialization_test_helper(dp, use_dill)
        it = iter(dp)
        _ = next(it)
        self._serialization_test_helper(dp, use_dill)
        _ = list(dp)
        self._serialization_test_helper(dp, use_dill)

    def test_serializable(self):
        if False:
            i = 10
            return i + 15
        picklable_datapipes: List = [(dp.map.Batcher, None, (2,), {}), (dp.map.Concater, None, (dp.map.SequenceWrapper(range(10)),), {}), (dp.map.Mapper, None, (), {}), (dp.map.Mapper, None, (_fake_fn,), {}), (dp.map.Mapper, None, (partial(_fake_add, 1),), {}), (dp.map.SequenceWrapper, range(10), (), {}), (dp.map.Shuffler, dp.map.SequenceWrapper([0] * 5), (), {}), (dp.map.Zipper, None, (dp.map.SequenceWrapper(range(10)),), {})]
        for (dpipe, custom_input, dp_args, dp_kwargs) in picklable_datapipes:
            if custom_input is None:
                custom_input = dp.map.SequenceWrapper(range(10))
            datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)
            self._serialization_test_for_single_dp(datapipe)

    def test_serializable_with_dill(self):
        if False:
            i = 10
            return i + 15
        'Only for DataPipes that take in a function as argument'
        input_dp = dp.map.SequenceWrapper(range(10))
        datapipes_with_lambda_fn: List[Tuple[Type[MapDataPipe], Tuple, Dict[str, Any]]] = [(dp.map.Mapper, (lambda_fn1,), {})]

        def _local_fns():
            if False:
                while True:
                    i = 10

            def _fn1(x):
                if False:
                    while True:
                        i = 10
                return x
            return _fn1
        fn1 = _local_fns()
        datapipes_with_local_fn: List[Tuple[Type[MapDataPipe], Tuple, Dict[str, Any]]] = [(dp.map.Mapper, (fn1,), {})]
        if HAS_DILL:
            for (dpipe, dp_args, dp_kwargs) in datapipes_with_lambda_fn + datapipes_with_local_fn:
                _ = dill.dumps(dpipe(input_dp, *dp_args, **dp_kwargs))
        else:
            msgs = ('^Lambda function is not supported by pickle', '^Local function is not supported by pickle')
            for (dps, msg) in zip((datapipes_with_lambda_fn, datapipes_with_local_fn), msgs):
                for (dpipe, dp_args, dp_kwargs) in dps:
                    with self.assertWarnsRegex(UserWarning, msg):
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)
                    with self.assertRaises((pickle.PicklingError, AttributeError)):
                        pickle.dumps(datapipe)

    def test_docstring(self):
        if False:
            print('Hello World!')
        '\n        Ensure functional form of MapDataPipe has the correct docstring from\n        the class form.\n\n        Regression test for https://github.com/pytorch/data/issues/792.\n        '
        input_dp = dp.map.SequenceWrapper(range(10))
        for dp_funcname in ['batch', 'concat', 'map', 'shuffle', 'zip']:
            if sys.version_info >= (3, 9):
                docstring = pydoc.render_doc(thing=getattr(input_dp, dp_funcname), forceload=True)
            elif sys.version_info < (3, 9):
                docstring = getattr(input_dp, dp_funcname).__doc__
            assert f'(functional name: ``{dp_funcname}``)' in docstring
            assert 'Args:' in docstring
            assert 'Example:' in docstring or 'Examples:' in docstring

    def test_sequence_wrapper_datapipe(self):
        if False:
            print('Hello World!')
        seq = list(range(10))
        input_dp = dp.map.SequenceWrapper(seq)
        self.assertEqual(seq, list(input_dp))
        seq.append(11)
        self.assertEqual(list(range(10)), list(input_dp))
        seq2 = [1, 2, 3]
        input_dp_non_deep = dp.map.SequenceWrapper(seq2, deepcopy=False)
        seq2.append(4)
        self.assertEqual(list(seq2), list(input_dp_non_deep))
        seq = list(range(10))
        n_elements_before_reset = 5
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(input_dp, n_elements_before_reset)
        self.assertEqual(list(range(5)), res_before_reset)
        self.assertEqual(seq, res_after_reset)
        self.assertEqual(len(seq), len(input_dp))

    def test_concat_mapdatapipe(self):
        if False:
            i = 10
            return i + 15
        input_dp1 = dp.map.SequenceWrapper(range(10))
        input_dp2 = dp.map.SequenceWrapper(range(5))
        with self.assertRaisesRegex(ValueError, 'Expected at least one DataPipe'):
            dp.map.Concater()
        with self.assertRaisesRegex(TypeError, 'Expected all inputs to be `MapDataPipe`'):
            dp.map.Concater(input_dp1, ())
        concat_dp = input_dp1.concat(input_dp2)
        self.assertEqual(len(concat_dp), 15)
        for index in range(15):
            self.assertEqual(concat_dp[index], (list(range(10)) + list(range(5)))[index])
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

    def test_zip_mapdatapipe(self):
        if False:
            i = 10
            return i + 15
        input_dp1 = dp.map.SequenceWrapper(range(10))
        input_dp2 = dp.map.SequenceWrapper(range(5))
        input_dp3 = dp.map.SequenceWrapper(range(15))
        with self.assertRaisesRegex(ValueError, 'Expected at least one DataPipe'):
            dp.map.Zipper()
        with self.assertRaisesRegex(TypeError, 'Expected all inputs to be `MapDataPipe`'):
            dp.map.Zipper(input_dp1, ())
        zip_dp = input_dp1.zip(input_dp2, input_dp3)
        self.assertEqual([(i, i, i) for i in range(5)], [zip_dp[i] for i in range(5)])
        with self.assertRaisesRegex(IndexError, 'out of range'):
            input_dp1.zip(input_dp2, input_dp3)[5]
        dp1 = dp.map.SequenceWrapper(range(10))
        shuffle_dp1 = dp1.batch(2)
        dp2 = dp.map.SequenceWrapper(range(10))
        shuffle_dp2 = dp2.batch(3)
        zip_dp1 = shuffle_dp1.zip(shuffle_dp2)
        self.assertEqual(4, len(list(zip_dp1)))
        zip_dp2 = shuffle_dp1.zip(dp2)
        self.assertEqual(5, len(list(zip_dp2)))
        zip_dp = input_dp1.zip(input_dp2, input_dp3)
        self.assertEqual(5, len(zip_dp))

    def test_shuffler_mapdatapipe(self):
        if False:
            return 10
        input_dp1 = dp.map.SequenceWrapper(range(10))
        input_dp2 = dp.map.SequenceWrapper({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
        shuffler_dp = input_dp1.shuffle()
        self.assertEqual(set(range(10)), set(shuffler_dp))
        shuffler_dp = input_dp2.shuffle(indices=['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(set(range(1, 6)), set(shuffler_dp))
        torch.manual_seed(123)
        shuffler_dp = input_dp1.shuffle()
        res = list(shuffler_dp)
        torch.manual_seed(123)
        self.assertEqual(list(shuffler_dp), res)
        shuffler_dp = input_dp1.shuffle().set_seed(123)
        res = list(shuffler_dp)
        shuffler_dp.set_seed(123)
        self.assertEqual(list(shuffler_dp), res)
        unshuffled_dp = input_dp1.shuffle().set_shuffle(False)
        self.assertEqual(list(unshuffled_dp), list(input_dp1))
        shuffler_dp = input_dp1.shuffle()
        n_elements_before_reset = 5
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(shuffler_dp, n_elements_before_reset)
        self.assertEqual(5, len(res_before_reset))
        for x in res_before_reset:
            self.assertTrue(x in set(range(10)))
        self.assertEqual(set(range(10)), set(res_after_reset))
        shuffler_dp = input_dp1.shuffle()
        self.assertEqual(10, len(shuffler_dp))
        from torch.utils.data.datapipes._hook_iterator import _SnapshotState
        shuffler_dp = input_dp1.shuffle()
        it = iter(shuffler_dp)
        for _ in range(2):
            next(it)
        shuffler_dp_copy = pickle.loads(pickle.dumps(shuffler_dp))
        exp = list(it)
        shuffler_dp_copy._snapshot_state = _SnapshotState.Restored
        self.assertEqual(exp, list(shuffler_dp_copy))

    def test_map_mapdatapipe(self):
        if False:
            i = 10
            return i + 15
        arr = range(10)
        input_dp = dp.map.SequenceWrapper(arr)

        def fn(item, dtype=torch.float, *, sum=False):
            if False:
                for i in range(10):
                    print('nop')
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()
        map_dp = input_dp.map(fn)
        self.assertEqual(len(input_dp), len(map_dp))
        for index in arr:
            self.assertEqual(map_dp[index], torch.tensor(input_dp[index], dtype=torch.float))
        map_dp = input_dp.map(partial(fn, dtype=torch.int, sum=True))
        self.assertEqual(len(input_dp), len(map_dp))
        for index in arr:
            self.assertEqual(map_dp[index], torch.tensor(input_dp[index], dtype=torch.int).sum())

    def test_batch_mapdatapipe(self):
        if False:
            for i in range(10):
                print('nop')
        arr = list(range(13))
        input_dp = dp.map.SequenceWrapper(arr)
        batch_dp = dp.map.Batcher(input_dp, batch_size=2)
        self.assertEqual([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12]], list(batch_dp))
        batch_dp = dp.map.Batcher(input_dp, batch_size=2, drop_last=True)
        self.assertEqual([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], list(batch_dp))
        batch_dp_2 = batch_dp.batch(batch_size=3)
        self.assertEqual([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], list(batch_dp_2))
        n_elements_before_reset = 3
        (res_before_reset, res_after_reset) = reset_after_n_next_calls(batch_dp, n_elements_before_reset)
        self.assertEqual([[0, 1], [2, 3], [4, 5]], res_before_reset)
        self.assertEqual([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], res_after_reset)
        self.assertEqual(6, len(batch_dp))
        self.assertEqual(2, len(batch_dp_2))
_generic_namedtuple_allowed = sys.version_info >= (3, 7) and sys.version_info < (3, 9)
if _generic_namedtuple_allowed:

    class InvalidData(Generic[T_co], NamedTuple):
        name: str
        data: T_co

class TestTyping(TestCase):

    def test_isinstance(self):
        if False:
            for i in range(10):
                print('nop')

        class A(IterDataPipe):
            pass

        class B(IterDataPipe):
            pass
        a = A()
        self.assertTrue(isinstance(a, A))
        self.assertFalse(isinstance(a, B))

    def test_protocol(self):
        if False:
            i = 10
            return i + 15
        try:
            from typing import Protocol
        except ImportError:
            from typing import _Protocol
            Protocol = _Protocol

        class P(Protocol):
            pass

        class A(IterDataPipe[P]):
            pass

    @skipTyping
    def test_subtype(self):
        if False:
            i = 10
            return i + 15
        from torch.utils.data.datapipes._typing import issubtype
        basic_type = (int, str, bool, float, complex, list, tuple, dict, set, T_co)
        for t in basic_type:
            self.assertTrue(issubtype(t, t))
            self.assertTrue(issubtype(t, Any))
            if t == T_co:
                self.assertTrue(issubtype(Any, t))
            else:
                self.assertFalse(issubtype(Any, t))
        for (t1, t2) in itertools.product(basic_type, basic_type):
            if t1 == t2 or t2 == T_co:
                self.assertTrue(issubtype(t1, t2))
            else:
                self.assertFalse(issubtype(t1, t2))
        T = TypeVar('T', int, str)
        S = TypeVar('S', bool, Union[str, int], Tuple[int, T])
        types = ((int, Optional[int]), (List, Union[int, list]), (Tuple[int, str], S), (Tuple[int, str], tuple), (T, S), (S, T_co), (T, Union[S, Set]))
        for (sub, par) in types:
            self.assertTrue(issubtype(sub, par))
            self.assertFalse(issubtype(par, sub))
        subscriptable_types = {List: 1, Tuple: 2, Set: 1, Dict: 2}
        for (subscript_type, n) in subscriptable_types.items():
            for ts in itertools.combinations(types, n):
                (subs, pars) = zip(*ts)
                sub = subscript_type[subs]
                par = subscript_type[pars]
                self.assertTrue(issubtype(sub, par))
                self.assertFalse(issubtype(par, sub))
                self.assertTrue(issubtype(par, sub, recursive=False))

    @skipTyping
    def test_issubinstance(self):
        if False:
            i = 10
            return i + 15
        from torch.utils.data.datapipes._typing import issubinstance
        basic_data = (1, '1', True, 1.0, complex(1.0, 0.0))
        basic_type = (int, str, bool, float, complex)
        S = TypeVar('S', bool, Union[str, int])
        for d in basic_data:
            self.assertTrue(issubinstance(d, Any))
            self.assertTrue(issubinstance(d, T_co))
            if type(d) in (bool, int, str):
                self.assertTrue(issubinstance(d, S))
            else:
                self.assertFalse(issubinstance(d, S))
            for t in basic_type:
                if type(d) == t:
                    self.assertTrue(issubinstance(d, t))
                else:
                    self.assertFalse(issubinstance(d, t))
        dt = (([1, '1', 2], List), (set({1, '1', 2}), Set))
        for (d, t) in dt:
            self.assertTrue(issubinstance(d, t))
            self.assertTrue(issubinstance(d, t[T_co]))
            self.assertFalse(issubinstance(d, t[int]))
        d = {'1': 1, '2': 2.0}
        self.assertTrue(issubinstance(d, Dict))
        self.assertTrue(issubinstance(d, Dict[str, T_co]))
        self.assertFalse(issubinstance(d, Dict[str, int]))
        d = (1, '1', 2)
        self.assertTrue(issubinstance(d, Tuple))
        self.assertTrue(issubinstance(d, Tuple[int, str, T_co]))
        self.assertFalse(issubinstance(d, Tuple[int, Any]))
        self.assertFalse(issubinstance(d, Tuple[int, int, int]))

    @skipTyping
    def test_compile_time(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(TypeError, "Expected 'Iterator' as the return"):

            class InvalidDP1(IterDataPipe[int]):

                def __iter__(self) -> str:
                    if False:
                        return 10
                    yield 0
        with self.assertRaisesRegex(TypeError, "Expected return type of '__iter__'"):

            class InvalidDP2(IterDataPipe[Tuple]):

                def __iter__(self) -> Iterator[int]:
                    if False:
                        for i in range(10):
                            print('nop')
                    yield 0
        with self.assertRaisesRegex(TypeError, "Expected return type of '__iter__'"):

            class InvalidDP3(IterDataPipe[Tuple[int, str]]):

                def __iter__(self) -> Iterator[tuple]:
                    if False:
                        i = 10
                        return i + 15
                    yield (0,)
        if _generic_namedtuple_allowed:
            with self.assertRaisesRegex(TypeError, 'is not supported by Python typing'):

                class InvalidDP4(IterDataPipe['InvalidData[int]']):
                    pass

        class DP1(IterDataPipe[Tuple[int, str]]):

            def __init__(self, length):
                if False:
                    return 10
                self.length = length

            def __iter__(self) -> Iterator[Tuple[int, str]]:
                if False:
                    while True:
                        i = 10
                for d in range(self.length):
                    yield (d, str(d))
        self.assertTrue(issubclass(DP1, IterDataPipe))
        dp1 = DP1(10)
        self.assertTrue(DP1.type.issubtype(dp1.type) and dp1.type.issubtype(DP1.type))
        dp1_ = DP1(5)
        self.assertEqual(dp1.type, dp1_.type)
        with self.assertRaisesRegex(TypeError, 'is not a generic class'):

            class InvalidDP5(DP1[tuple]):

                def __iter__(self) -> Iterator[tuple]:
                    if False:
                        for i in range(10):
                            print('nop')
                    yield (0,)

        class DP2(IterDataPipe[T_co]):

            def __iter__(self) -> Iterator[T_co]:
                if False:
                    while True:
                        i = 10
                yield from range(10)
        self.assertTrue(issubclass(DP2, IterDataPipe))
        dp2 = DP2()
        self.assertTrue(DP2.type.issubtype(dp2.type) and dp2.type.issubtype(DP2.type))
        dp2_ = DP2()
        self.assertEqual(dp2.type, dp2_.type)

        class DP3(IterDataPipe[Tuple[T_co, str]]):
            """ DataPipe without fixed type with __init__ function"""

            def __init__(self, datasource):
                if False:
                    for i in range(10):
                        print('nop')
                self.datasource = datasource

            def __iter__(self) -> Iterator[Tuple[T_co, str]]:
                if False:
                    while True:
                        i = 10
                for d in self.datasource:
                    yield (d, str(d))
        self.assertTrue(issubclass(DP3, IterDataPipe))
        dp3 = DP3(range(10))
        self.assertTrue(DP3.type.issubtype(dp3.type) and dp3.type.issubtype(DP3.type))
        dp3_ = DP3(5)
        self.assertEqual(dp3.type, dp3_.type)

        class DP4(IterDataPipe[tuple]):
            """ DataPipe without __iter__ annotation"""

            def __iter__(self):
                if False:
                    return 10
                raise NotImplementedError
        self.assertTrue(issubclass(DP4, IterDataPipe))
        dp4 = DP4()
        self.assertTrue(dp4.type.param == tuple)

        class DP5(IterDataPipe):
            """ DataPipe without type annotation"""

            def __iter__(self) -> Iterator[str]:
                if False:
                    for i in range(10):
                        print('nop')
                raise NotImplementedError
        self.assertTrue(issubclass(DP5, IterDataPipe))
        dp5 = DP5()
        from torch.utils.data.datapipes._typing import issubtype
        self.assertTrue(issubtype(dp5.type.param, Any) and issubtype(Any, dp5.type.param))

        class DP6(IterDataPipe[int]):
            """ DataPipe with plain Iterator"""

            def __iter__(self) -> Iterator:
                if False:
                    print('Hello World!')
                raise NotImplementedError
        self.assertTrue(issubclass(DP6, IterDataPipe))
        dp6 = DP6()
        self.assertTrue(dp6.type.param == int)

        class DP7(IterDataPipe[Awaitable[T_co]]):
            """ DataPipe with abstract base class"""
        self.assertTrue(issubclass(DP7, IterDataPipe))
        self.assertTrue(DP7.type.param == Awaitable[T_co])

        class DP8(DP7[str]):
            """ DataPipe subclass from a DataPipe with abc type"""
        self.assertTrue(issubclass(DP8, IterDataPipe))
        self.assertTrue(DP8.type.param == Awaitable[str])

    @skipTyping
    def test_construct_time(self):
        if False:
            for i in range(10):
                print('nop')

        class DP0(IterDataPipe[Tuple]):

            @argument_validation
            def __init__(self, dp: IterDataPipe):
                if False:
                    print('Hello World!')
                self.dp = dp

            def __iter__(self) -> Iterator[Tuple]:
                if False:
                    return 10
                for d in self.dp:
                    yield (d, str(d))

        class DP1(IterDataPipe[int]):

            @argument_validation
            def __init__(self, dp: IterDataPipe[Tuple[int, str]]):
                if False:
                    for i in range(10):
                        print('nop')
                self.dp = dp

            def __iter__(self) -> Iterator[int]:
                if False:
                    print('Hello World!')
                for (a, b) in self.dp:
                    yield a
        datasource = [(1, '1'), (2, '2'), (3, '3')]
        with self.assertRaisesRegex(TypeError, "Expected argument 'dp' as a IterDataPipe"):
            dp0 = DP0(datasource)
        dp0 = DP0(dp.iter.IterableWrapper(range(10)))
        with self.assertRaisesRegex(TypeError, "Expected type of argument 'dp' as a subtype"):
            dp1 = DP1(dp0)

    @skipTyping
    def test_runtime(self):
        if False:
            for i in range(10):
                print('nop')

        class DP(IterDataPipe[Tuple[int, T_co]]):

            def __init__(self, datasource):
                if False:
                    for i in range(10):
                        print('nop')
                self.ds = datasource

            @runtime_validation
            def __iter__(self) -> Iterator[Tuple[int, T_co]]:
                if False:
                    print('Hello World!')
                yield from self.ds
        dss = ([(1, '1'), (2, '2')], [(1, 1), (2, '2')])
        for ds in dss:
            dp0 = DP(ds)
            self.assertEqual(list(dp0), ds)
            self.assertEqual(list(dp0), ds)
        dss = ([(1, 1), ('2', 2)], [[1, '1'], [2, '2']], [1, '1', 2, '2'])
        for ds in dss:
            dp0 = DP(ds)
            with self.assertRaisesRegex(RuntimeError, 'Expected an instance as subtype'):
                list(dp0)
            with runtime_validation_disabled():
                self.assertEqual(list(dp0), ds)
                with runtime_validation_disabled():
                    self.assertEqual(list(dp0), ds)
            with self.assertRaisesRegex(RuntimeError, 'Expected an instance as subtype'):
                list(dp0)

    @skipTyping
    def test_reinforce(self):
        if False:
            for i in range(10):
                print('nop')
        T = TypeVar('T', int, str)

        class DP(IterDataPipe[T]):

            def __init__(self, ds):
                if False:
                    i = 10
                    return i + 15
                self.ds = ds

            @runtime_validation
            def __iter__(self) -> Iterator[T]:
                if False:
                    while True:
                        i = 10
                yield from self.ds
        ds = list(range(10))
        dp0 = DP(ds).reinforce_type(int)
        self.assertTrue(dp0.type, int)
        self.assertEqual(list(dp0), ds)
        with self.assertRaisesRegex(TypeError, "'expected_type' must be a type"):
            dp1 = DP(ds).reinforce_type(1)
        with self.assertRaisesRegex(TypeError, "Expected 'expected_type' as subtype of"):
            dp2 = DP(ds).reinforce_type(float)
        dp3 = DP(ds).reinforce_type(str)
        with self.assertRaisesRegex(RuntimeError, 'Expected an instance as subtype'):
            list(dp3)
        with runtime_validation_disabled():
            self.assertEqual(list(dp3), ds)

class NumbersDataset(IterDataPipe):

    def __init__(self, size=10):
        if False:
            print('Hello World!')
        self.size = size

    def __iter__(self):
        if False:
            while True:
                i = 10
        yield from range(self.size)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.size

class TestGraph(TestCase):

    class CustomIterDataPipe(IterDataPipe):

        def add_v(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return x + self.v

        def __init__(self, source_dp, v=1):
            if False:
                i = 10
                return i + 15
            self._dp = source_dp.map(self.add_v)
            self.v = 1

        def __iter__(self):
            if False:
                return 10
            yield from self._dp

        def __hash__(self):
            if False:
                print('Hello World!')
            raise NotImplementedError

    def test_simple_traverse(self):
        if False:
            i = 10
            return i + 15
        numbers_dp = NumbersDataset(size=50)
        shuffled_dp = numbers_dp.shuffle()
        sharded_dp = shuffled_dp.sharding_filter()
        mapped_dp = sharded_dp.map(lambda x: x * 10)
        graph = traverse_dps(mapped_dp)
        expected: Dict[Any, Any] = {id(mapped_dp): (mapped_dp, {id(sharded_dp): (sharded_dp, {id(shuffled_dp): (shuffled_dp, {id(numbers_dp): (numbers_dp, {})})})})}
        self.assertEqual(expected, graph)
        dps = torch.utils.data.graph_settings.get_all_graph_pipes(graph)
        self.assertEqual(len(dps), 4)
        for datapipe in (numbers_dp, shuffled_dp, sharded_dp, mapped_dp):
            self.assertTrue(datapipe in dps)

    def test_traverse_forked(self):
        if False:
            while True:
                i = 10
        numbers_dp = NumbersDataset(size=50)
        (dp0, dp1, dp2) = numbers_dp.fork(num_instances=3)
        dp0_upd = dp0.map(lambda x: x * 10)
        dp1_upd = dp1.filter(lambda x: x % 3 == 1)
        combined_dp = dp0_upd.mux(dp1_upd, dp2)
        graph = traverse_dps(combined_dp)
        expected = {id(combined_dp): (combined_dp, {id(dp0_upd): (dp0_upd, {id(dp0): (dp0, {id(dp0.main_datapipe): (dp0.main_datapipe, {id(dp0.main_datapipe.main_datapipe): (dp0.main_datapipe.main_datapipe, {})})})}), id(dp1_upd): (dp1_upd, {id(dp1): (dp1, {id(dp1.main_datapipe): (dp1.main_datapipe, {id(dp1.main_datapipe.main_datapipe): (dp1.main_datapipe.main_datapipe, {})})})}), id(dp2): (dp2, {id(dp2.main_datapipe): (dp2.main_datapipe, {id(dp2.main_datapipe.main_datapipe): (dp2.main_datapipe.main_datapipe, {})})})})}
        self.assertEqual(expected, graph)
        dps = torch.utils.data.graph_settings.get_all_graph_pipes(graph)
        self.assertEqual(len(dps), 8)
        for _dp in [numbers_dp, dp0.main_datapipe, dp0, dp1, dp2, dp0_upd, dp1_upd, combined_dp]:
            self.assertTrue(_dp in dps)

    def test_traverse_mapdatapipe(self):
        if False:
            i = 10
            return i + 15
        source_dp = dp.map.SequenceWrapper(range(10))
        map_dp = source_dp.map(partial(_fake_add, 1))
        graph = traverse_dps(map_dp)
        expected: Dict[Any, Any] = {id(map_dp): (map_dp, {id(source_dp): (source_dp, {})})}
        self.assertEqual(expected, graph)

    def test_traverse_mixdatapipe(self):
        if False:
            while True:
                i = 10
        source_map_dp = dp.map.SequenceWrapper(range(10))
        iter_dp = dp.iter.IterableWrapper(source_map_dp)
        graph = traverse_dps(iter_dp)
        expected: Dict[Any, Any] = {id(iter_dp): (iter_dp, {id(source_map_dp): (source_map_dp, {})})}
        self.assertEqual(expected, graph)

    def test_traverse_circular_datapipe(self):
        if False:
            i = 10
            return i + 15
        source_iter_dp = dp.iter.IterableWrapper(list(range(10)))
        circular_dp = TestGraph.CustomIterDataPipe(source_iter_dp)
        graph = traverse_dps(circular_dp)
        expected: Dict[Any, Any] = {id(circular_dp): (circular_dp, {id(circular_dp._dp): (circular_dp._dp, {id(source_iter_dp): (source_iter_dp, {})})})}
        self.assertEqual(expected, graph)
        dps = torch.utils.data.graph_settings.get_all_graph_pipes(graph)
        self.assertEqual(len(dps), 3)
        for _dp in [circular_dp, circular_dp._dp, source_iter_dp]:
            self.assertTrue(_dp in dps)

    def test_traverse_unhashable_datapipe(self):
        if False:
            for i in range(10):
                print('nop')
        source_iter_dp = dp.iter.IterableWrapper(list(range(10)))
        unhashable_dp = TestGraph.CustomIterDataPipe(source_iter_dp)
        graph = traverse_dps(unhashable_dp)
        with self.assertRaises(NotImplementedError):
            hash(unhashable_dp)
        expected: Dict[Any, Any] = {id(unhashable_dp): (unhashable_dp, {id(unhashable_dp._dp): (unhashable_dp._dp, {id(source_iter_dp): (source_iter_dp, {})})})}
        self.assertEqual(expected, graph)

def unbatch(x):
    if False:
        for i in range(10):
            print('nop')
    return x[0]

class TestSerialization(TestCase):

    @skipIfNoDill
    def test_spawn_lambdas_iter(self):
        if False:
            i = 10
            return i + 15
        idp = dp.iter.IterableWrapper(range(3)).map(lambda x: x + 1).shuffle()
        dl = DataLoader(idp, num_workers=2, shuffle=True, multiprocessing_context='spawn', collate_fn=unbatch, batch_size=1)
        result = list(dl)
        self.assertEqual([1, 1, 2, 2, 3, 3], sorted(result))

    @skipIfNoDill
    def test_spawn_lambdas_map(self):
        if False:
            print('Hello World!')
        mdp = dp.map.SequenceWrapper(range(3)).map(lambda x: x + 1).shuffle()
        dl = DataLoader(mdp, num_workers=2, shuffle=True, multiprocessing_context='spawn', collate_fn=unbatch, batch_size=1)
        result = list(dl)
        self.assertEqual([1, 1, 2, 2, 3, 3], sorted(result))

class TestCircularSerialization(TestCase):

    class CustomIterDataPipe(IterDataPipe):

        @staticmethod
        def add_one(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 1

        @classmethod
        def classify(cls, x):
            if False:
                i = 10
                return i + 15
            return 0

        def add_v(self, x):
            if False:
                return 10
            return x + self.v

        def __init__(self, fn, source_dp=None):
            if False:
                i = 10
                return i + 15
            self.fn = fn
            self.source_dp = source_dp if source_dp else dp.iter.IterableWrapper([1, 2, 4])
            self._dp = self.source_dp.map(self.add_one).map(self.add_v).demux(2, self.classify)[0]
            self.v = 1

        def __iter__(self):
            if False:
                i = 10
                return i + 15
            yield from self._dp

    def test_circular_serialization_with_pickle(self):
        if False:
            while True:
                i = 10
        dp1 = TestCircularSerialization.CustomIterDataPipe(fn=_fake_fn)
        self.assertTrue(list(dp1) == list(pickle.loads(pickle.dumps(dp1))))
        child_1 = dp1._dp
        dm_1 = child_1.main_datapipe
        m2_1 = dm_1.main_datapipe
        m1_1 = m2_1.datapipe
        src_1 = m1_1.datapipe
        res1 = traverse_dps(dp1)
        exp_res_1 = {id(dp1): (dp1, {id(src_1): (src_1, {}), id(child_1): (child_1, {id(dm_1): (dm_1, {id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})})})})}
        self.assertEqual(res1, exp_res_1)
        dp2 = TestCircularSerialization.CustomIterDataPipe(fn=_fake_fn, source_dp=dp1)
        self.assertTrue(list(dp2) == list(pickle.loads(pickle.dumps(dp2))))
        child_2 = dp2._dp
        dm_2 = child_2.main_datapipe
        m2_2 = dm_2.main_datapipe
        m1_2 = m2_2.datapipe
        res2 = traverse_dps(dp2)
        exp_res_2 = {id(dp2): (dp2, {id(dp1): (dp1, {id(src_1): (src_1, {}), id(child_1): (child_1, {id(dm_1): (dm_1, {id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})})})}), id(child_2): (child_2, {id(dm_2): (dm_2, {id(m2_2): (m2_2, {id(m1_2): (m1_2, {id(dp1): (dp1, {id(src_1): (src_1, {}), id(child_1): (child_1, {id(dm_1): (dm_1, {id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})})})})})})})})})}
        self.assertEqual(res2, exp_res_2)

    class LambdaIterDataPipe(CustomIterDataPipe):

        def __init__(self, fn, source_dp=None):
            if False:
                print('Hello World!')
            super().__init__(fn, source_dp)
            self.container = [lambda x: x + 1]
            self.lambda_fn = lambda x: x + 1
            self._dp = self.source_dp.map(self.add_one).map(self.lambda_fn).map(self.add_v).demux(2, self.classify)[0]

    @skipIfNoDill
    @skipIf(True, 'Dill Tests')
    def test_circular_serialization_with_dill(self):
        if False:
            return 10
        dp1 = TestCircularSerialization.LambdaIterDataPipe(lambda x: x + 1)
        self.assertTrue(list(dp1) == list(dill.loads(dill.dumps(dp1))))
        child_1 = dp1._dp
        dm_1 = child_1.main_datapipe
        m2_1 = dm_1.main_datapipe
        m1_1 = m2_1.datapipe
        src_1 = m1_1.datapipe
        res1 = traverse_dps(dp1)
        exp_res_1 = {id(dp1): (dp1, {id(src_1): (src_1, {}), id(child_1): (child_1, {id(dm_1): (dm_1, {id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})})})})}
        self.assertEqual(res1, exp_res_1)
        dp2 = TestCircularSerialization.LambdaIterDataPipe(fn=_fake_fn, source_dp=dp1)
        self.assertTrue(list(dp2) == list(dill.loads(dill.dumps(dp2))))
        child_2 = dp2._dp
        dm_2 = child_2.main_datapipe
        m2_2 = dm_2.main_datapipe
        m1_2 = m2_2.datapipe
        res2 = traverse_dps(dp2)
        exp_res_2 = {id(dp2): (dp2, {id(dp1): (dp1, {id(src_1): (src_1, {}), id(child_1): (child_1, {id(dm_1): (dm_1, {id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})})})}), id(child_2): (child_2, {id(dm_2): (dm_2, {id(m2_2): (m2_2, {id(m1_2): (m1_2, {id(dp1): (dp1, {id(src_1): (src_1, {}), id(child_1): (child_1, {id(dm_1): (dm_1, {id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})})})})})})})})})}
        self.assertEqual(res2, exp_res_2)

class CustomShardingIterDataPipe(IterDataPipe):

    def __init__(self, dp):
        if False:
            for i in range(10):
                print('nop')
        self.dp = dp
        self.num_of_instances = 1
        self.instance_id = 0

    def apply_sharding(self, num_of_instances, instance_id):
        if False:
            while True:
                i = 10
        self.num_of_instances = num_of_instances
        self.instance_id = instance_id

    def __iter__(self):
        if False:
            return 10
        for (i, d) in enumerate(self.dp):
            if i % self.num_of_instances == self.instance_id:
                yield d

class TestSharding(TestCase):

    def _get_pipeline(self):
        if False:
            i = 10
            return i + 15
        numbers_dp = NumbersDataset(size=10)
        (dp0, dp1) = numbers_dp.fork(num_instances=2)
        dp0_upd = dp0.map(_mul_10)
        dp1_upd = dp1.filter(_mod_3_test)
        combined_dp = dp0_upd.mux(dp1_upd)
        return combined_dp

    def _get_dill_pipeline(self):
        if False:
            for i in range(10):
                print('nop')
        numbers_dp = NumbersDataset(size=10)
        (dp0, dp1) = numbers_dp.fork(num_instances=2)
        dp0_upd = dp0.map(lambda x: x * 10)
        dp1_upd = dp1.filter(lambda x: x % 3 == 1)
        combined_dp = dp0_upd.mux(dp1_upd)
        return combined_dp

    def test_simple_sharding(self):
        if False:
            print('Hello World!')
        sharded_dp = self._get_pipeline().sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 1)
        items = list(sharded_dp)
        self.assertEqual([1, 20], items)
        all_items = [0, 1, 10, 4, 20, 7]
        items = []
        for i in range(3):
            sharded_dp = self._get_pipeline().sharding_filter()
            torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, i)
            items += list(sharded_dp)
        self.assertEqual(sorted(all_items), sorted(items))

    def test_sharding_groups(self):
        if False:
            while True:
                i = 10

        def construct_sharded_pipe():
            if False:
                print('Hello World!')
            sharding_pipes = []
            dp = NumbersDataset(size=90)
            dp = dp.sharding_filter(sharding_group_filter=SHARDING_PRIORITIES.DISTRIBUTED)
            sharding_pipes.append(dp)
            dp = dp.sharding_filter(sharding_group_filter=SHARDING_PRIORITIES.MULTIPROCESSING)
            sharding_pipes.append(dp)
            dp = dp.sharding_filter(sharding_group_filter=300)
            sharding_pipes.append(dp)
            return (dp, sharding_pipes)
        (dp, sharding_pipes) = construct_sharded_pipe()
        for pipe in sharding_pipes:
            pipe.apply_sharding(2, 1, sharding_group=SHARDING_PRIORITIES.DISTRIBUTED)
            pipe.apply_sharding(5, 3, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING)
            pipe.apply_sharding(3, 1, sharding_group=300)
        actual = list(dp)
        expected = [17, 47, 77]
        self.assertEqual(expected, actual)
        self.assertEqual(3, len(dp))
        (dp, _) = construct_sharded_pipe()
        dp.apply_sharding(2, 1, sharding_group=SHARDING_PRIORITIES.DEFAULT)
        with self.assertRaises(Exception):
            dp.apply_sharding(5, 3, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING)
        (dp, _) = construct_sharded_pipe()
        dp.apply_sharding(5, 3, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING)
        with self.assertRaises(Exception):
            dp.apply_sharding(2, 1, sharding_group=SHARDING_PRIORITIES.DEFAULT)

    def test_sharding_groups_in_legacy_grouping_package(self):
        if False:
            while True:
                i = 10
        with self.assertWarnsRegex(FutureWarning, 'Please use `SHARDING_PRIORITIES` from the `torch.utils.data.datapipes.iter.sharding`'):
            from torch.utils.data.datapipes.iter.grouping import SHARDING_PRIORITIES as LEGACY_SHARDING_PRIORITIES

        def construct_sharded_pipe():
            if False:
                for i in range(10):
                    print('nop')
            sharding_pipes = []
            dp = NumbersDataset(size=90)
            dp = dp.sharding_filter(sharding_group_filter=LEGACY_SHARDING_PRIORITIES.DISTRIBUTED)
            sharding_pipes.append(dp)
            dp = dp.sharding_filter(sharding_group_filter=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING)
            sharding_pipes.append(dp)
            dp = dp.sharding_filter(sharding_group_filter=300)
            sharding_pipes.append(dp)
            return (dp, sharding_pipes)
        (dp, sharding_pipes) = construct_sharded_pipe()
        for pipe in sharding_pipes:
            pipe.apply_sharding(2, 1, sharding_group=LEGACY_SHARDING_PRIORITIES.DISTRIBUTED)
            pipe.apply_sharding(5, 3, sharding_group=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING)
            pipe.apply_sharding(3, 1, sharding_group=300)
        actual = list(dp)
        expected = [17, 47, 77]
        self.assertEqual(expected, actual)
        self.assertEqual(3, len(dp))
        (dp, _) = construct_sharded_pipe()
        dp.apply_sharding(2, 1, sharding_group=LEGACY_SHARDING_PRIORITIES.DEFAULT)
        with self.assertRaises(Exception):
            dp.apply_sharding(5, 3, sharding_group=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING)
        (dp, _) = construct_sharded_pipe()
        dp.apply_sharding(5, 3, sharding_group=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING)
        with self.assertRaises(Exception):
            dp.apply_sharding(2, 1, sharding_group=LEGACY_SHARDING_PRIORITIES.DEFAULT)

    def test_legacy_custom_sharding(self):
        if False:
            for i in range(10):
                print('nop')
        dp = self._get_pipeline()
        sharded_dp = CustomShardingIterDataPipe(dp)
        torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 1)
        items = list(sharded_dp)
        self.assertEqual([1, 20], items)

    def test_sharding_length(self):
        if False:
            for i in range(10):
                print('nop')
        numbers_dp = dp.iter.IterableWrapper(range(13))
        sharded_dp0 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp0, 3, 0)
        sharded_dp1 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp1, 3, 1)
        sharded_dp2 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp2, 3, 2)
        self.assertEqual(13, len(numbers_dp))
        self.assertEqual(5, len(sharded_dp0))
        self.assertEqual(4, len(sharded_dp1))
        self.assertEqual(4, len(sharded_dp2))
        numbers_dp = dp.iter.IterableWrapper(range(1))
        sharded_dp0 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp0, 2, 0)
        sharded_dp1 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp1, 2, 1)
        self.assertEqual(1, len(sharded_dp0))
        self.assertEqual(0, len(sharded_dp1))

    def test_old_dataloader(self):
        if False:
            for i in range(10):
                print('nop')
        dp0 = self._get_pipeline()
        expected = list(dp0)
        dp0 = self._get_pipeline().sharding_filter()
        dl = DataLoader(dp0, batch_size=1, shuffle=False, num_workers=2)
        items = []
        for i in dl:
            items.append(i)
        self.assertEqual(sorted(expected), sorted(items))

    def test_legacy_custom_sharding_with_old_dataloader(self):
        if False:
            for i in range(10):
                print('nop')
        dp0 = self._get_pipeline()
        expected = list(dp0)
        dp0 = self._get_pipeline()
        dp0 = CustomShardingIterDataPipe(dp0)
        dl = DataLoader(dp0, batch_size=1, shuffle=False, num_workers=2)
        items = []
        for i in dl:
            items.append(i)
        self.assertEqual(sorted(expected), sorted(items))

    def test_multi_sharding(self):
        if False:
            i = 10
            return i + 15
        numbers_dp = dp.iter.IterableWrapper(range(13))
        sharded_dp = numbers_dp.sharding_filter()
        sharded_dp = sharded_dp.sharding_filter()
        with self.assertRaisesRegex(RuntimeError, 'Sharding twice on a single pipeline'):
            torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 0)
        numbers_dp = dp.iter.IterableWrapper(range(13)).sharding_filter()
        (dp1, dp2) = numbers_dp.fork(2)
        sharded_dp = dp1.sharding_filter()
        zip_dp = dp2.zip(sharded_dp)
        with self.assertRaisesRegex(RuntimeError, 'Sharding twice on a single pipeline'):
            torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)
        numbers_dp = dp.iter.IterableWrapper(range(13))
        (dp1, dp2) = numbers_dp.fork(2)
        sharded_dp = dp1.sharding_filter()
        zip_dp = dp2.zip(sharded_dp).sharding_filter()
        with self.assertRaisesRegex(RuntimeError, 'Sharding twice on a single pipeline'):
            torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)
        numbers_dp = dp.iter.IterableWrapper(range(13)).sharding_filter()
        (dp1, dp2) = numbers_dp.fork(2)
        zip_dp = dp1.zip(dp2)
        torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)
        self.assertEqual(list(zip_dp), [(i * 3, i * 3) for i in range(13 // 3 + 1)])
        numbers_dp = dp.iter.IterableWrapper(range(13))
        (dp1, dp2) = numbers_dp.fork(2)
        sharded_dp1 = dp1.sharding_filter()
        sharded_dp2 = dp2.sharding_filter()
        zip_dp = sharded_dp1.zip(sharded_dp2)
        torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)
        self.assertEqual(list(zip_dp), [(i * 3, i * 3) for i in range(13 // 3 + 1)])

class TestIterDataPipeSingletonConstraint(TestCase):
    """
    Each `IterDataPipe` can only have one active iterator. Whenever a new iterator is created, older
    iterators are invalidated. These tests aim to ensure `IterDataPipe` follows this behavior.
    """

    def _check_single_iterator_invalidation_logic(self, source_dp: IterDataPipe):
        if False:
            print('Hello World!')
        '\n        Given a IterDataPipe, verifies that the iterator can be read, reset, and the creation of\n        a second iterator invalidates the first one.\n        '
        it1 = iter(source_dp)
        self.assertEqual(list(range(10)), list(it1))
        it1 = iter(source_dp)
        self.assertEqual(list(range(10)), list(it1))
        it1 = iter(source_dp)
        self.assertEqual(0, next(it1))
        it2 = iter(source_dp)
        self.assertEqual(0, next(it2))
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it1)

    def test_iterdatapipe_singleton_generator(self):
        if False:
            i = 10
            return i + 15
        "\n        Testing for the case where IterDataPipe's `__iter__` is a generator function.\n        "
        source_dp: IterDataPipe = dp.iter.IterableWrapper(range(10))
        self._check_single_iterator_invalidation_logic(source_dp)
        dps = source_dp.map(_fake_fn).filter(_fake_filter_fn)
        self._check_single_iterator_invalidation_logic(dps)
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            for _ in zip(source_dp, source_dp):
                pass
        for _ in zip(list(source_dp), list(source_dp)):
            pass

    def test_iterdatapipe_singleton_self_next(self):
        if False:
            while True:
                i = 10
        "\n        Testing for the case where IterDataPipe's `__iter__` returns `self` and there is a `__next__` method\n        Note that the following DataPipe by is singleton by default (because `__iter__` returns `self`).\n        "

        class _CustomIterDP_Self(IterDataPipe):

            def __init__(self, iterable):
                if False:
                    print('Hello World!')
                self.source = iterable
                self.iterable = iter(iterable)

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                self.reset()
                return self

            def __next__(self):
                if False:
                    return 10
                return next(self.iterable)

            def reset(self):
                if False:
                    print('Hello World!')
                self.iterable = iter(self.source)
        source_dp = _CustomIterDP_Self(range(10))
        res = list(source_dp)
        it = iter(source_dp)
        self.assertEqual(res, list(it))
        source_dp = _CustomIterDP_Self(range(10))
        self._check_single_iterator_invalidation_logic(source_dp)
        self.assertEqual(1, next(source_dp))
        source_dp = _CustomIterDP_Self(dp.iter.IterableWrapper(range(10)).map(_fake_fn).filter(_fake_filter_fn))
        self._check_single_iterator_invalidation_logic(source_dp)
        self.assertEqual(1, next(source_dp))
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            for _ in zip(source_dp, source_dp):
                pass

    def test_iterdatapipe_singleton_new_object(self):
        if False:
            i = 10
            return i + 15
        "\n        Testing for the case where IterDataPipe's `__iter__` isn't a generator nor returns `self`,\n        and there isn't a `__next__` method.\n        "

        class _CustomIterDP(IterDataPipe):

            def __init__(self, iterable):
                if False:
                    while True:
                        i = 10
                self.iterable = iter(iterable)

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                return self.iterable
        source_dp = _CustomIterDP(range(10))
        it1 = iter(source_dp)
        self.assertEqual(0, next(it1))
        it2 = iter(source_dp)
        self.assertEqual(1, next(it2))
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it1)
        source_dp = _CustomIterDP(dp.iter.IterableWrapper(range(10)).map(_fake_fn).filter(_fake_filter_fn))
        it1 = iter(source_dp)
        self.assertEqual(0, next(it1))
        it2 = iter(source_dp)
        self.assertEqual(1, next(it2))
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it1)
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            for _ in zip(source_dp, source_dp):
                pass

    def test_iterdatapipe_singleton_buggy(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Buggy test case case where IterDataPipe's `__iter__` returns a new object, but also has\n        a `__next__` method.\n        "

        class _CustomIterDP(IterDataPipe):

            def __init__(self, iterable):
                if False:
                    print('Hello World!')
                self.source = iterable
                self.iterable = iter(iterable)

            def __iter__(self):
                if False:
                    i = 10
                    return i + 15
                return iter(self.source)

            def __next__(self):
                if False:
                    i = 10
                    return i + 15
                return next(self.iterable)
        source_dp = _CustomIterDP(range(10))
        self._check_single_iterator_invalidation_logic(source_dp)
        self.assertEqual(0, next(source_dp))
        source_dp = _CustomIterDP(range(10))
        self.assertEqual(0, next(source_dp))
        it1 = iter(source_dp)
        self.assertEqual(0, next(it1))
        self.assertEqual(1, next(source_dp))
        it2 = iter(source_dp)
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it1)
        self.assertEqual(2, next(source_dp))
        self.assertEqual(list(range(10)), list(it2))

    def test_iterdatapipe_singleton_constraint_multiple_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Testing for the case where IterDataPipe has multiple child DataPipes as outputs.\n        '
        source_dp: IterDataPipe = dp.iter.IterableWrapper(range(10))
        (cdp1, cdp2) = source_dp.fork(num_instances=2)
        (it1, it2) = (iter(cdp1), iter(cdp2))
        self.assertEqual(list(range(10)), list(it1))
        self.assertEqual(list(range(10)), list(it2))
        (it1, it2) = (iter(cdp1), iter(cdp2))
        with warnings.catch_warnings(record=True) as wa:
            it3 = iter(cdp1)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), 'child DataPipes are not exhausted')
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it1)
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it2)
        self.assertEqual(0, next(it3))
        it4 = iter(cdp2)
        self.assertEqual(1, next(it3))
        self.assertEqual(list(range(10)), list(it4))
        source_dp = dp.iter.IterableWrapper(range(10))
        (cdp1, cdp2) = source_dp.fork(num_instances=2)
        (it1, it2) = (iter(cdp1), iter(cdp2))
        self.assertEqual(list(range(10)), list(it1))
        self.assertEqual(list(range(10)), list(it2))
        (it1, it2) = (iter(cdp1), iter(cdp2))
        self.assertEqual(0, next(it1))
        self.assertEqual(0, next(it2))
        it3 = iter(source_dp)
        self.assertEqual(0, next(it3))
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it1)
        self.assertEqual(1, next(it3))
        source_dp = dp.iter.IterableWrapper(range(10)).map(_fake_fn).filter(_fake_filter_fn)
        (cdp1, cdp2) = source_dp.fork(num_instances=2)
        (it1, it2) = (iter(cdp1), iter(cdp2))
        self.assertEqual(list(range(10)), list(it1))
        self.assertEqual(list(range(10)), list(it2))
        (it1, it2) = (iter(cdp1), iter(cdp2))
        with warnings.catch_warnings(record=True) as wa:
            it3 = iter(cdp1)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), 'child DataPipes are not exhausted')
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it1)
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it2)
        with warnings.catch_warnings(record=True) as wa:
            (it1, it2) = (iter(cdp1), iter(cdp2))
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), 'child DataPipes are not exhausted')
        self.assertEqual(0, next(it1))
        self.assertEqual(0, next(it2))
        it3 = iter(source_dp)
        self.assertEqual(0, next(it3))
        with self.assertRaisesRegex(RuntimeError, 'This iterator has been invalidated'):
            next(it1)
        self.assertEqual(1, next(it3))

class TestIterDataPipeCountSampleYielded(TestCase):

    def _yield_count_test_helper(self, datapipe, n_expected_samples):
        if False:
            for i in range(10):
                print('nop')
        res = list(datapipe)
        self.assertEqual(len(res), datapipe._number_of_samples_yielded)
        it = iter(datapipe)
        res = []
        for (i, value) in enumerate(it):
            res.append(value)
            if i == n_expected_samples - 1:
                break
        self.assertEqual(n_expected_samples, datapipe._number_of_samples_yielded)
        it = iter(datapipe)
        res = list(it)
        self.assertEqual(len(res), datapipe._number_of_samples_yielded)

    def test_iterdatapipe_sample_yielded_generator_function(self):
        if False:
            return 10
        datapipe: IterDataPipe = dp.iter.IterableWrapper(range(10))
        self._yield_count_test_helper(datapipe, n_expected_samples=5)

    def test_iterdatapipe_sample_yielded_generator_function_exception(self):
        if False:
            print('Hello World!')

        class _CustomGeneratorFnDataPipe(IterDataPipe):

            def __iter__(self):
                if False:
                    print('Hello World!')
                yield 0
                yield 1
                yield 2
                raise RuntimeError('Custom test error after yielding 3 elements')
                yield 3
        datapipe: IterDataPipe = _CustomGeneratorFnDataPipe()
        with self.assertRaisesRegex(RuntimeError, 'Custom test error after yielding 3 elements'):
            list(datapipe)
        self.assertEqual(3, datapipe._number_of_samples_yielded)
        it = iter(datapipe)
        with self.assertRaisesRegex(RuntimeError, 'Custom test error after yielding 3 elements'):
            list(it)
        self.assertEqual(3, datapipe._number_of_samples_yielded)

    def test_iterdatapipe_sample_yielded_return_self(self):
        if False:
            print('Hello World!')

        class _CustomGeneratorDataPipe(IterDataPipe):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.source = iter(range(10))

            def __iter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.source

            def reset(self):
                if False:
                    print('Hello World!')
                self.source = iter(range(10))
        datapipe: IterDataPipe = _CustomGeneratorDataPipe()
        self._yield_count_test_helper(datapipe, n_expected_samples=5)

    def test_iterdatapipe_sample_yielded_next(self):
        if False:
            return 10

        class _CustomNextDataPipe(IterDataPipe):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.source = iter(range(10))

            def __iter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self

            def __next__(self):
                if False:
                    return 10
                return next(self.source)

            def reset(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.source = iter(range(10))
        datapipe: IterDataPipe = _CustomNextDataPipe()
        self._yield_count_test_helper(datapipe, n_expected_samples=5)

    def test_iterdatapipe_sample_yielded_next_exception(self):
        if False:
            for i in range(10):
                print('nop')

        class _CustomNextDataPipe(IterDataPipe):

            def __init__(self):
                if False:
                    return 10
                self.source = iter(range(10))
                self.count = 0

            def __iter__(self):
                if False:
                    i = 10
                    return i + 15
                return self

            def __next__(self):
                if False:
                    for i in range(10):
                        print('nop')
                if self.count == 3:
                    raise RuntimeError('Custom test error after yielding 3 elements')
                self.count += 1
                return next(self.source)

            def reset(self):
                if False:
                    while True:
                        i = 10
                self.count = 0
                self.source = iter(range(10))
        datapipe: IterDataPipe = _CustomNextDataPipe()
        with self.assertRaisesRegex(RuntimeError, 'Custom test error after yielding 3 elements'):
            list(datapipe)
        self.assertEqual(3, datapipe._number_of_samples_yielded)
        it = iter(datapipe)
        with self.assertRaisesRegex(RuntimeError, 'Custom test error after yielding 3 elements'):
            list(it)
        self.assertEqual(3, datapipe._number_of_samples_yielded)

class _CustomNonGeneratorTestDataPipe(IterDataPipe):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.n = 10
        self.source = list(range(self.n))

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.source)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.n

class _CustomSelfNextTestDataPipe(IterDataPipe):

    def __init__(self):
        if False:
            return 10
        self.n = 10
        self.iter = iter(range(self.n))

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        return next(self.iter)

    def reset(self):
        if False:
            print('Hello World!')
        self.iter = iter(range(self.n))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.n

class TestIterDataPipeGraphFastForward(TestCase):

    def _fast_forward_graph_test_helper(self, datapipe, fast_forward_fn, expected_res, n_iterations=3, rng=None):
        if False:
            print('Hello World!')
        if rng is None:
            rng = torch.Generator()
        rng = rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(datapipe, rng)
        rng.manual_seed(0)
        fast_forward_fn(datapipe, n_iterations, rng)
        actual_res = list(datapipe)
        self.assertEqual(len(datapipe) - n_iterations, len(actual_res))
        self.assertEqual(expected_res[n_iterations:], actual_res)
        rng.manual_seed(0)
        fast_forward_fn(datapipe, n_iterations, rng)
        it = iter(datapipe)
        actual_res = list(it)
        self.assertEqual(len(datapipe) - n_iterations, len(actual_res))
        self.assertEqual(expected_res[n_iterations:], actual_res)
        with self.assertRaises(StopIteration):
            next(it)

    def test_simple_snapshot_graph(self):
        if False:
            while True:
                i = 10
        graph1 = dp.iter.IterableWrapper(range(10))
        res1 = list(range(10))
        self._fast_forward_graph_test_helper(graph1, _simple_graph_snapshot_restoration, expected_res=res1)
        graph2 = graph1.map(_mul_10)
        res2 = [10 * x for x in res1]
        self._fast_forward_graph_test_helper(graph2, _simple_graph_snapshot_restoration, expected_res=res2)
        rng = torch.Generator()
        graph3 = graph2.shuffle()
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(graph3, rng)
        res3 = list(graph3)
        self._fast_forward_graph_test_helper(graph3, _simple_graph_snapshot_restoration, expected_res=res3)
        graph4 = graph3.map(_mul_10)
        res4 = [10 * x for x in res3]
        self._fast_forward_graph_test_helper(graph4, _simple_graph_snapshot_restoration, expected_res=res4)
        batch_size = 2
        graph5 = graph4.batch(batch_size)
        res5 = [res4[i:i + batch_size] for i in range(0, len(res4), batch_size)]
        self._fast_forward_graph_test_helper(graph5, _simple_graph_snapshot_restoration, expected_res=res5)
        (cdp1, cdp2) = graph5.fork(2)
        graph6 = cdp1.zip(cdp2)
        rng = rng.manual_seed(100)
        torch.utils.data.graph_settings.apply_random_seed(graph6, rng)
        res6 = [(x, x) for x in res5]
        self._fast_forward_graph_test_helper(graph6, _simple_graph_snapshot_restoration, expected_res=res6)
        graph7 = cdp1.concat(cdp2)
        res7 = res5 * 2
        self._fast_forward_graph_test_helper(graph7, _simple_graph_snapshot_restoration, expected_res=res7)
        with self.assertRaisesRegex(RuntimeError, 'Snapshot restoration cannot be applied.'):
            _simple_graph_snapshot_restoration(graph7, 1)
            _simple_graph_snapshot_restoration(graph7, 1)

    def test_simple_snapshot_custom_non_generator(self):
        if False:
            i = 10
            return i + 15
        graph = _CustomNonGeneratorTestDataPipe()
        self._fast_forward_graph_test_helper(graph, _simple_graph_snapshot_restoration, expected_res=range(10))

    def test_simple_snapshot_custom_self_next(self):
        if False:
            for i in range(10):
                print('nop')
        graph = _CustomSelfNextTestDataPipe()
        self._fast_forward_graph_test_helper(graph, _simple_graph_snapshot_restoration, expected_res=range(10))

    def _snapshot_test_helper(self, datapipe, expected_res, n_iter=3, rng=None):
        if False:
            return 10
        '\n        Extend the previous test with serialization and deserialization test.\n        '
        if rng is None:
            rng = torch.Generator()
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(datapipe, rng)
        it = iter(datapipe)
        for _ in range(n_iter):
            next(it)
        serialized_graph = pickle.dumps(datapipe)
        deserialized_graph = pickle.loads(serialized_graph)
        self.assertEqual(n_iter, datapipe._number_of_samples_yielded)
        self.assertEqual(n_iter, deserialized_graph._number_of_samples_yielded)
        rng_for_deserialized = torch.Generator()
        rng_for_deserialized.manual_seed(0)
        _simple_graph_snapshot_restoration(deserialized_graph, n_iter, rng=rng_for_deserialized)
        self.assertEqual(expected_res[n_iter:], list(it))
        self.assertEqual(expected_res[n_iter:], list(deserialized_graph))

    def test_simple_snapshot_graph_with_serialization(self):
        if False:
            while True:
                i = 10
        graph1 = dp.iter.IterableWrapper(range(10))
        res1 = list(range(10))
        self._snapshot_test_helper(graph1, expected_res=res1)
        graph2 = graph1.map(_mul_10)
        res2 = [10 * x for x in res1]
        self._snapshot_test_helper(graph2, expected_res=res2)
        rng = torch.Generator()
        graph3 = graph2.shuffle()
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(graph3, rng)
        res3 = list(graph3)
        self._snapshot_test_helper(graph3, expected_res=res3)
        graph4 = graph3.map(_mul_10)
        res4 = [10 * x for x in res3]
        self._snapshot_test_helper(graph4, expected_res=res4)
        batch_size = 2
        graph5 = graph4.batch(batch_size)
        res5 = [res4[i:i + batch_size] for i in range(0, len(res4), batch_size)]
        self._snapshot_test_helper(graph5, expected_res=res5)
        (cdp1, cdp2) = graph5.fork(2)
        graph6 = cdp1.zip(cdp2)
        res6 = [(x, x) for x in res5]
        self._snapshot_test_helper(graph6, expected_res=res6)
        graph7 = cdp1.concat(cdp2)
        res7 = res5 * 2
        self._snapshot_test_helper(graph7, expected_res=res7)

    def test_simple_snapshot_graph_repeated(self):
        if False:
            return 10
        (cdp1, cdp2) = dp.iter.IterableWrapper(range(10)).map(_mul_10).shuffle().map(_mul_10).map(_mul_10).fork(2)
        graph = cdp1.zip(cdp2)
        rng = torch.Generator()
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(graph, rng)
        expected_res = list(graph)
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(graph, rng)
        it = iter(graph)
        n_iter = 3
        for _ in range(n_iter):
            next(it)
        serialized_graph = pickle.dumps(graph)
        deserialized_graph = pickle.loads(serialized_graph)
        rng_for_deserialized = torch.Generator()
        rng_for_deserialized.manual_seed(0)
        _simple_graph_snapshot_restoration(deserialized_graph, deserialized_graph._number_of_samples_yielded, rng=rng_for_deserialized)
        it = iter(deserialized_graph)
        self.assertEqual(expected_res[3], next(it))
        serialized_graph2 = pickle.dumps(deserialized_graph)
        deserialized_graph2 = pickle.loads(serialized_graph2)
        rng_for_deserialized = torch.Generator()
        rng_for_deserialized.manual_seed(0)
        _simple_graph_snapshot_restoration(deserialized_graph2, deserialized_graph._number_of_samples_yielded, rng=rng_for_deserialized)
        self.assertEqual(expected_res[4:], list(deserialized_graph2))
if __name__ == '__main__':
    run_tests()
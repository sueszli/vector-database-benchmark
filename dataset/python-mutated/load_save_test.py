import hypothesis.strategies as st
from hypothesis import given, assume, settings
import io
import math
import numpy as np
import os
import struct
import unittest
from pathlib import Path
from typing import Dict, Generator, List, NamedTuple, Optional, Tuple, Type
from caffe2.proto import caffe2_pb2
from caffe2.proto.caffe2_pb2 import BlobSerializationOptions
from caffe2.python import core, test_util, workspace
if workspace.has_gpu_support:
    DEVICES = [caffe2_pb2.CPU, workspace.GpuDeviceType]
    max_gpuid = workspace.NumGpuDevices() - 1
else:
    DEVICES = [caffe2_pb2.CPU]
    max_gpuid = 0

class MiniDBEntry(NamedTuple):
    key: str
    value_size: int

class TestLoadSaveBase(test_util.TestCase):

    def __init__(self, methodName, db_type='minidb'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(methodName)
        self._db_type = db_type

    @settings(deadline=None)
    @given(src_device_type=st.sampled_from(DEVICES), src_gpu_id=st.integers(min_value=0, max_value=max_gpuid), dst_device_type=st.sampled_from(DEVICES), dst_gpu_id=st.integers(min_value=0, max_value=max_gpuid))
    def load_save(self, src_device_type, src_gpu_id, dst_device_type, dst_gpu_id):
        if False:
            return 10
        workspace.ResetWorkspace()
        dtypes = [np.float16, np.float32, np.float64, bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16]
        arrays = [np.random.permutation(6).reshape(2, 3).astype(T) for T in dtypes]
        assume(core.IsGPUDeviceType(src_device_type) or src_gpu_id == 0)
        assume(core.IsGPUDeviceType(dst_device_type) or dst_gpu_id == 0)
        src_device_option = core.DeviceOption(src_device_type, src_gpu_id)
        dst_device_option = core.DeviceOption(dst_device_type, dst_gpu_id)
        for (i, arr) in enumerate(arrays):
            self.assertTrue(workspace.FeedBlob(str(i), arr, src_device_option))
            self.assertTrue(workspace.HasBlob(str(i)))
        tmp_folder = self.make_tempdir()
        op = core.CreateOperator('Save', [str(i) for i in range(len(arrays))], [], absolute_path=1, db=str(tmp_folder / 'db'), db_type=self._db_type)
        self.assertTrue(workspace.RunOperatorOnce(op))
        workspace.ResetWorkspace()
        self.assertEqual(len(workspace.Blobs()), 0)

        def _LoadTest(keep_device, device_type, gpu_id, blobs, loadAll):
            if False:
                while True:
                    i = 10
            'A helper subfunction to test keep and not keep.'
            op = core.CreateOperator('Load', [], blobs, absolute_path=1, db=str(tmp_folder / 'db'), db_type=self._db_type, device_option=dst_device_option, keep_device=keep_device, load_all=loadAll)
            self.assertTrue(workspace.RunOperatorOnce(op))
            for (i, arr) in enumerate(arrays):
                self.assertTrue(workspace.HasBlob(str(i)))
                fetched = workspace.FetchBlob(str(i))
                self.assertEqual(fetched.dtype, arr.dtype)
                np.testing.assert_array_equal(workspace.FetchBlob(str(i)), arr)
                proto = caffe2_pb2.BlobProto()
                proto.ParseFromString(workspace.SerializeBlob(str(i)))
                self.assertTrue(proto.HasField('tensor'))
                self.assertEqual(proto.tensor.device_detail.device_type, device_type)
                if core.IsGPUDeviceType(device_type):
                    self.assertEqual(proto.tensor.device_detail.device_id, gpu_id)
        blobs = [str(i) for i in range(len(arrays))]
        _LoadTest(1, src_device_type, src_gpu_id, blobs, 0)
        _LoadTest(0, dst_device_type, dst_gpu_id, blobs, 0)
        _LoadTest(1, src_device_type, src_gpu_id, blobs, 0)
        workspace.ResetWorkspace()
        _LoadTest(0, dst_device_type, dst_gpu_id, blobs, 0)
        workspace.ResetWorkspace()
        _LoadTest(1, src_device_type, src_gpu_id, [], 1)
        _LoadTest(1, src_device_type, src_gpu_id, [], 1)
        _LoadTest(0, dst_device_type, dst_gpu_id, [], 1)
        workspace.ResetWorkspace()
        _LoadTest(0, dst_device_type, dst_gpu_id, [], 1)
        workspace.ResetWorkspace()
        _LoadTest(1, src_device_type, src_gpu_id, blobs, 1)
        workspace.ResetWorkspace()
        _LoadTest(0, dst_device_type, dst_gpu_id, blobs, 1)

    def saveFile(self, tmp_folder: Path, db_name: str, db_type: str, start_blob_id: int) -> Tuple[str, List[np.ndarray]]:
        if False:
            while True:
                i = 10
        dtypes = [np.float16, np.float32, np.float64, bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16]
        arrays = [np.random.permutation(6).reshape(2, 3).astype(T) for T in dtypes]
        for (i, arr) in enumerate(arrays):
            self.assertTrue(workspace.FeedBlob(str(i + start_blob_id), arr))
            self.assertTrue(workspace.HasBlob(str(i + start_blob_id)))
        tmp_file = str(tmp_folder / db_name)
        op = core.CreateOperator('Save', [str(i + start_blob_id) for i in range(len(arrays))], [], absolute_path=1, db=tmp_file, db_type=db_type)
        workspace.RunOperatorOnce(op)
        return (tmp_file, arrays)

class TestLoadSave(TestLoadSaveBase):

    def testLoadSave(self):
        if False:
            while True:
                i = 10
        self.load_save()

    def testRepeatedArgs(self):
        if False:
            while True:
                i = 10
        dtypes = [np.float16, np.float32, np.float64, bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16]
        arrays = [np.random.permutation(6).reshape(2, 3).astype(T) for T in dtypes]
        for (i, arr) in enumerate(arrays):
            self.assertTrue(workspace.FeedBlob(str(i), arr))
            self.assertTrue(workspace.HasBlob(str(i)))
        tmp_folder = self.make_tempdir()
        op = core.CreateOperator('Save', [str(i) for i in range(len(arrays))] * 2, [], absolute_path=1, db=str(tmp_folder / 'db'), db_type=self._db_type)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

    def testLoadExcessblobs(self):
        if False:
            while True:
                i = 10
        tmp_folder = self.make_tempdir()
        (tmp_file, arrays) = self.saveFile(tmp_folder, 'db', self._db_type, 0)
        op = core.CreateOperator('Load', [], [str(i) for i in range(len(arrays))] * 2, absolute_path=1, db=tmp_file, db_type=self._db_type, load_all=False)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)
        op = core.CreateOperator('Load', [], [str(len(arrays) + i) for i in [-1, 0]], absolute_path=1, db=tmp_file, db_type=self._db_type, load_all=True)
        with self.assertRaises(RuntimeError):
            workspace.ResetWorkspace()
            workspace.RunOperatorOnce(op)
        op = core.CreateOperator('Load', [], [str(len(arrays) + i) for i in range(2)], absolute_path=1, db=tmp_file, db_type=self._db_type, load_all=True)
        with self.assertRaises(RuntimeError):
            workspace.ResetWorkspace()
            workspace.RunOperatorOnce(op)

    def testTruncatedFile(self):
        if False:
            i = 10
            return i + 15
        tmp_folder = self.make_tempdir()
        (tmp_file, arrays) = self.saveFile(tmp_folder, 'db', self._db_type, 0)
        with open(tmp_file, 'wb+') as fdest:
            fdest.seek(20, os.SEEK_END)
            fdest.truncate()
        op = core.CreateOperator('Load', [], [str(i) for i in range(len(arrays))], absolute_path=1, db=tmp_file, db_type=self._db_type, load_all=False)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)
        op = core.CreateOperator('Load', [], [], absolute_path=1, db=tmp_file, db_type=self._db_type, load_all=True)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

    def testBlobNameOverrides(self):
        if False:
            print('Hello World!')
        original_names = ['blob_a', 'blob_b', 'blob_c']
        new_names = ['x', 'y', 'z']
        blobs = [np.random.permutation(6) for i in range(3)]
        for (i, blob) in enumerate(blobs):
            self.assertTrue(workspace.FeedBlob(original_names[i], blob))
            self.assertTrue(workspace.HasBlob(original_names[i]))
        self.assertEqual(len(workspace.Blobs()), 3)
        tmp_folder = self.make_tempdir()
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(core.CreateOperator('Save', original_names, [], absolute_path=1, strip_prefix='.temp', blob_name_overrides=new_names, db=str(tmp_folder / 'db'), db_type=self._db_type))
        self.assertTrue(workspace.RunOperatorOnce(core.CreateOperator('Save', original_names, [], absolute_path=1, blob_name_overrides=new_names, db=str(tmp_folder / 'db'), db_type=self._db_type)))
        self.assertTrue(workspace.ResetWorkspace())
        self.assertEqual(len(workspace.Blobs()), 0)
        self.assertTrue(workspace.RunOperatorOnce(core.CreateOperator('Load', [], [], absolute_path=1, db=str(tmp_folder / 'db'), db_type=self._db_type, load_all=1)))
        self.assertEqual(len(workspace.Blobs()), 3)
        for (i, name) in enumerate(new_names):
            self.assertTrue(workspace.HasBlob(name))
            self.assertTrue((workspace.FetchBlob(name) == blobs[i]).all())
        load_new_names = ['blob_x', 'blob_y', 'blob_z']
        self.assertTrue(workspace.RunOperatorOnce(core.CreateOperator('Load', [], load_new_names[0:1], absolute_path=1, db=str(tmp_folder / 'db'), db_type=self._db_type, source_blob_names=new_names[0:1])))
        self.assertEqual(len(workspace.Blobs()), 4)
        for (i, name) in enumerate(load_new_names[0:1]):
            self.assertTrue(workspace.HasBlob(name))
            self.assertTrue((workspace.FetchBlob(name) == blobs[i]).all())
        self.assertTrue(workspace.RunOperatorOnce(core.CreateOperator('Load', [], load_new_names[0:3], absolute_path=1, db=str(tmp_folder / 'db'), db_type=self._db_type, source_blob_names=new_names[0:3])))
        self.assertEqual(len(workspace.Blobs()), 6)
        for (i, name) in enumerate(load_new_names[0:3]):
            self.assertTrue(workspace.HasBlob(name))
            self.assertTrue((workspace.FetchBlob(name) == blobs[i]).all())

    def testMissingFile(self):
        if False:
            while True:
                i = 10
        tmp_folder = self.make_tempdir()
        tmp_file = tmp_folder / 'missing_db'
        op = core.CreateOperator('Load', [], [], absolute_path=1, db=str(tmp_file), db_type=self._db_type, load_all=True)
        with self.assertRaises(RuntimeError):
            try:
                workspace.RunOperatorOnce(op)
            except RuntimeError as e:
                print(e)
                raise

    def testLoadMultipleFilesGivenSourceBlobNames(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_folder = self.make_tempdir()
        (db_file_1, arrays_1) = self.saveFile(tmp_folder, 'db1', self._db_type, 0)
        (db_file_2, arrays_2) = self.saveFile(tmp_folder, 'db2', self._db_type, len(arrays_1))
        db_files = [db_file_1, db_file_2]
        blobs_names = [str(i) for i in range(len(arrays_1) + len(arrays_2))]
        workspace.ResetWorkspace()
        self.assertEqual(len(workspace.Blobs()), 0)
        self.assertTrue(workspace.RunOperatorOnce(core.CreateOperator('Load', [], blobs_names, absolute_path=1, dbs=db_files, db_type=self._db_type, source_blob_names=blobs_names)))
        self.assertEqual(len(workspace.Blobs()), len(blobs_names))
        for i in range(len(arrays_1)):
            np.testing.assert_array_equal(workspace.FetchBlob(str(i)), arrays_1[i])
        for i in range(len(arrays_2)):
            np.testing.assert_array_equal(workspace.FetchBlob(str(i + len(arrays_1))), arrays_2[i])

    def testLoadAllMultipleFiles(self):
        if False:
            i = 10
            return i + 15
        tmp_folder = self.make_tempdir()
        (db_file_1, arrays_1) = self.saveFile(tmp_folder, 'db1', self._db_type, 0)
        (db_file_2, arrays_2) = self.saveFile(tmp_folder, 'db2', self._db_type, len(arrays_1))
        db_files = [db_file_1, db_file_2]
        workspace.ResetWorkspace()
        self.assertEqual(len(workspace.Blobs()), 0)
        self.assertTrue(workspace.RunOperatorOnce(core.CreateOperator('Load', [], [], absolute_path=1, dbs=db_files, db_type=self._db_type, load_all=True)))
        self.assertEqual(len(workspace.Blobs()), len(arrays_1) + len(arrays_2))
        for i in range(len(arrays_1)):
            np.testing.assert_array_equal(workspace.FetchBlob(str(i)), arrays_1[i])
        for i in range(len(arrays_2)):
            np.testing.assert_array_equal(workspace.FetchBlob(str(i + len(arrays_1))), arrays_2[i])

    def testLoadAllMultipleFilesWithSameKey(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_folder = self.make_tempdir()
        (db_file_1, arrays_1) = self.saveFile(tmp_folder, 'db1', self._db_type, 0)
        (db_file_2, arrays_2) = self.saveFile(tmp_folder, 'db2', self._db_type, 0)
        db_files = [db_file_1, db_file_2]
        workspace.ResetWorkspace()
        self.assertEqual(len(workspace.Blobs()), 0)
        op = core.CreateOperator('Load', [], [], absolute_path=1, dbs=db_files, db_type=self._db_type, load_all=True)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

    def testLoadRepeatedFiles(self):
        if False:
            return 10
        tmp_folder = self.make_tempdir()
        (tmp_file, arrays) = self.saveFile(tmp_folder, 'db', self._db_type, 0)
        db_files = [tmp_file, tmp_file]
        workspace.ResetWorkspace()
        self.assertEqual(len(workspace.Blobs()), 0)
        op = core.CreateOperator('Load', [], [str(i) for i in range(len(arrays))], absolute_path=1, dbs=db_files, db_type=self._db_type, load_all=False)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

    def testLoadWithDBOptions(self) -> None:
        if False:
            return 10
        tmp_folder = self.make_tempdir()
        (tmp_file, arrays) = self.saveFile(tmp_folder, 'db', self._db_type, 0)
        db_files = [tmp_file, tmp_file]
        workspace.ResetWorkspace()
        self.assertEqual(len(workspace.Blobs()), 0)
        db_options = b'test_db_options'
        op = core.CreateOperator('Load', [], [str(i) for i in range(len(arrays))], absolute_path=1, dbs=db_files, db_type=self._db_type, load_all=False, db_options=db_options)
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(op)

    def create_test_blobs(self, size: int=1234, feed: bool=True) -> List[Tuple[str, np.ndarray]]:
        if False:
            while True:
                i = 10

        def int_array(dtype: Type[np.integer], size: int) -> np.ndarray:
            if False:
                return 10
            info = np.iinfo(dtype)
            return np.random.randint(info.min, info.max, size, dtype=dtype)

        def float_array(dtype: Type[np.floating], size: int) -> np.ndarray:
            if False:
                while True:
                    i = 10
            return np.random.random_sample(size).astype(dtype)
        blobs = [('int8_data', int_array(np.int8, size)), ('int16_data', int_array(np.int16, size)), ('int32_data', int_array(np.int32, size)), ('int64_data', int_array(np.int64, size)), ('uint8_data', int_array(np.uint8, size)), ('uint16_data', int_array(np.uint16, size)), ('float16_data', float_array(np.float16, size)), ('float32_data', float_array(np.float32, size)), ('float64_data', float_array(np.float64, size))]
        if feed:
            for (name, data) in blobs:
                workspace.FeedBlob(name, data)
        return blobs

    def load_blobs(self, blob_names: List[str], dbs: List[str], db_type: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        workspace.ResetWorkspace()
        self.assertEqual(len(workspace.Blobs()), 0)
        load_op = core.CreateOperator('Load', [], blob_names, absolute_path=1, dbs=dbs, db_type=db_type or self._db_type)
        self.assertTrue(workspace.RunOperatorOnce(load_op))
        self.assertEqual(len(workspace.Blobs()), len(blob_names))

    def load_and_check_blobs(self, blobs: List[Tuple[str, np.ndarray]], dbs: List[str], db_type: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        self.load_blobs([name for (name, data) in blobs], dbs, db_type)
        for (name, data) in blobs:
            np.testing.assert_array_equal(workspace.FetchBlob(name), data)

    def _read_minidb_entries(self, path: Path) -> Generator[MiniDBEntry, None, None]:
        if False:
            print('Hello World!')
        'Read the entry information out of a minidb file.\n        '
        header = struct.Struct('=ii')
        with path.open('rb') as f:
            while True:
                buf = f.read(header.size)
                if not buf:
                    break
                if len(buf) < header.size:
                    raise Exception('early EOF in minidb header')
                (key_len, value_len) = header.unpack(buf)
                if key_len < 0 or value_len < 0:
                    raise Exception(f'invalid minidb header: ({key_len}, {value_len})')
                key = f.read(key_len)
                if len(key) < key_len:
                    raise Exception('early EOF in minidb key')
                f.seek(value_len, io.SEEK_CUR)
                yield MiniDBEntry(key=key.decode('utf-8'), value_size=value_len)

    def _read_chunk_info(self, path: Path) -> Dict[str, List[MiniDBEntry]]:
        if False:
            for i in range(10):
                print('nop')
        'Read a minidb file and return the names of each blob and how many\n        chunks are stored for that blob.\n        '
        chunk_id_separator = '#%'
        results: Dict[str, List[MiniDBEntry]] = {}
        for entry in self._read_minidb_entries(path):
            parts = entry.key.rsplit(chunk_id_separator, 1)
            if len(parts) == 0:
                assert entry.key not in results
                results[entry.key] = [entry]
            else:
                blob_name = parts[0]
                results.setdefault(blob_name, [])
                results[blob_name].append(entry)
        return results

    def _test_save_with_chunk_size(self, num_elems: int, chunk_size: int, expected_num_chunks: int) -> None:
        if False:
            i = 10
            return i + 15
        tmp_folder = self.make_tempdir()
        tmp_file = str(tmp_folder / 'save.output')
        blobs = self.create_test_blobs(num_elems)
        save_op = core.CreateOperator('Save', [name for (name, data) in blobs], [], absolute_path=1, db=tmp_file, db_type=self._db_type, chunk_size=chunk_size)
        self.assertTrue(workspace.RunOperatorOnce(save_op))
        self.load_and_check_blobs(blobs, [tmp_file])
        blob_chunks = self._read_chunk_info(Path(tmp_file))
        for (blob_name, chunks) in blob_chunks.items():
            self.assertEqual(len(chunks), expected_num_chunks)

    def testSaveWithChunkSize(self) -> None:
        if False:
            return 10
        num_elems = 1234
        chunk_size = 32
        expected_num_chunks = math.ceil(num_elems / chunk_size)
        self._test_save_with_chunk_size(num_elems=num_elems, chunk_size=chunk_size, expected_num_chunks=expected_num_chunks)

    def testSaveWithDefaultChunkSize(self) -> None:
        if False:
            print('Hello World!')
        default_chunk_size = 1000000
        self._test_save_with_chunk_size(num_elems=default_chunk_size + 10, chunk_size=-1, expected_num_chunks=2)

    def testSaveWithNoChunking(self) -> None:
        if False:
            return 10
        default_chunk_size = 1000000
        self._test_save_with_chunk_size(num_elems=default_chunk_size + 10, chunk_size=0, expected_num_chunks=1)

    def testSaveWithOptions(self) -> None:
        if False:
            print('Hello World!')
        tmp_folder = self.make_tempdir()
        tmp_file = str(tmp_folder / 'save.output')
        num_elems = 1234
        blobs = self.create_test_blobs(num_elems)
        save_op = core.CreateOperator('Save', [name for (name, data) in blobs], [], absolute_path=1, db=tmp_file, db_type=self._db_type, chunk_size=40, options=caffe2_pb2.SerializationOptions(options=[BlobSerializationOptions(blob_name_regex='int16_data', chunk_size=10), BlobSerializationOptions(blob_name_regex='.*16_data', chunk_size=20), BlobSerializationOptions(blob_name_regex='float16_data', chunk_size=30)]))
        self.assertTrue(workspace.RunOperatorOnce(save_op))
        self.load_and_check_blobs(blobs, [tmp_file])
        blob_chunks = self._read_chunk_info(Path(tmp_file))
        self.assertEqual(len(blob_chunks['int16_data']), math.ceil(num_elems / 10))
        self.assertEqual(len(blob_chunks['uint16_data']), math.ceil(num_elems / 20))
        self.assertEqual(len(blob_chunks['float16_data']), math.ceil(num_elems / 20))
        self.assertEqual(len(blob_chunks['int64_data']), math.ceil(num_elems / 40))

    def testSaveWithDBOptions(self) -> None:
        if False:
            i = 10
            return i + 15
        num_elems = 1234
        chunk_size = 32
        expected_num_chunks = math.ceil(num_elems / chunk_size)
        tmp_folder = self.make_tempdir()
        tmp_file = str(tmp_folder / 'save.output')
        blobs = self.create_test_blobs(num_elems)
        db_options = b'test_db_options'
        save_op = core.CreateOperator('Save', [name for (name, data) in blobs], [], absolute_path=1, db=tmp_file, db_type=self._db_type, chunk_size=chunk_size, db_options=db_options)
        self.assertTrue(workspace.RunOperatorOnce(save_op))
        self.load_and_check_blobs(blobs, [tmp_file])
        blob_chunks = self._read_chunk_info(Path(tmp_file))
        for (blob_name, chunks) in blob_chunks.items():
            self.assertEqual(len(chunks), expected_num_chunks)

    def testSaveFloatToBfloat16(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        tmp_folder = self.make_tempdir()
        tmp_file = str(tmp_folder / 'save.output')
        float_data = np.random.random_sample(4000).astype(np.float32)
        workspace.FeedBlob('float1', float_data)
        workspace.FeedBlob('float2', float_data)
        blob_names = ['float1', 'float2']
        save_op = core.CreateOperator('Save', blob_names, [], absolute_path=1, db=tmp_file, db_type=self._db_type, options=caffe2_pb2.SerializationOptions(options=[BlobSerializationOptions(blob_name_regex='float1', float_format=BlobSerializationOptions.FLOAT_BFLOAT16)]))
        self.assertTrue(workspace.RunOperatorOnce(save_op))
        if workspace.has_fbgemm:
            blob_chunks = self._read_chunk_info(Path(tmp_file))
            self.assertEqual(len(blob_chunks['float1']), 1, blob_chunks['float1'])
            self.assertEqual(len(blob_chunks['float2']), 1, blob_chunks['float2'])
            self.assertLess(blob_chunks['float1'][0].value_size, 0.6 * blob_chunks['float2'][0].value_size)
        self.load_blobs(blob_names, [tmp_file])
        np.testing.assert_array_equal(workspace.FetchBlob('float2'), float_data)
        np.testing.assert_array_almost_equal(workspace.FetchBlob('float1'), float_data, decimal=2)

    def testEstimateBlobSizes(self) -> None:
        if False:
            print('Hello World!')
        float_data = np.random.random_sample(4000).astype(np.float32)
        workspace.FeedBlob('float1', float_data)
        workspace.FeedBlob('float2', float_data)
        workspace.FeedBlob('float3', np.random.random_sample(2).astype(np.float32))
        workspace.FeedBlob('ui16', np.random.randint(0, 65535, size=1024, dtype=np.uint16))
        options = caffe2_pb2.SerializationOptions(options=[BlobSerializationOptions(blob_name_regex='float1', float_format=BlobSerializationOptions.FLOAT_BFLOAT16, chunk_size=500)])
        get_blobs_op = core.CreateOperator('EstimateAllBlobSizes', [], ['blob_names', 'blob_sizes'], options=options)
        self.assertTrue(workspace.RunOperatorOnce(get_blobs_op))
        blob_names = workspace.FetchBlob('blob_names')
        blob_sizes = workspace.FetchBlob('blob_sizes')
        sizes_by_name: Dict[str, int] = {}
        for (idx, name) in enumerate(blob_names):
            sizes_by_name[name.decode('utf-8')] = blob_sizes[idx]
        expected_blobs = ['float1', 'float2', 'float3', 'ui16', 'blob_names', 'blob_sizes']
        self.assertEqual(set(sizes_by_name.keys()), set(expected_blobs))

        def check_expected_blob_size(name: str, num_elems: int, elem_size: int, num_chunks: int=1) -> None:
            if False:
                i = 10
                return i + 15
            per_chunk_overhead = 50
            expected_size = num_chunks * (len(name) + per_chunk_overhead) + num_elems * elem_size
            self.assertEqual(sizes_by_name[name], expected_size, f'expected size mismatch for {name}')
        check_expected_blob_size('ui16', 1024, 3)
        check_expected_blob_size('float2', 4000, 4)
        check_expected_blob_size('float3', 2, 4)
        float1_num_chunks = 4000 // 500
        if workspace.has_fbgemm:
            check_expected_blob_size('float1', 4000, 2, float1_num_chunks)
        else:
            check_expected_blob_size('float1', 4000, 4, float1_num_chunks)
        check_expected_blob_size('blob_names', len(expected_blobs), 50)
        check_expected_blob_size('blob_sizes', len(expected_blobs), 8)
        tmp_folder = self.make_tempdir()
        tmp_file = str(tmp_folder / 'save.output')
        save_op = core.CreateOperator('Save', list(sizes_by_name.keys()), [], absolute_path=1, db=tmp_file, db_type=self._db_type, options=options)
        self.assertTrue(workspace.RunOperatorOnce(save_op))
        blob_chunks = self._read_chunk_info(Path(tmp_file))
        saved_sizes: Dict[str, int] = {}
        for (blob_name, chunks) in blob_chunks.items():
            total_size = sum((chunk.value_size for chunk in chunks))
            saved_sizes[blob_name] = total_size
        for name in expected_blobs:
            estimated_size = sizes_by_name[name]
            saved_size = saved_sizes[name]
            difference = abs(estimated_size - saved_size)
            error_pct = 100.0 * (difference / saved_size)
            print(f'{name}: estimated={estimated_size} actual={saved_size} error={error_pct:.2f}%')
            if name == 'blob_names':
                continue
            if difference > 100:
                self.assertLess(error_pct, 25.0)
if __name__ == '__main__':
    unittest.main()
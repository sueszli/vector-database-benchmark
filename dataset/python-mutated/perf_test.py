"""Tests for the MongoDB Driver Performance Benchmarking Spec."""
from __future__ import annotations
import multiprocessing as mp
import os
import sys
import tempfile
import time
import warnings
from typing import Any, List
try:
    import simplejson as json
except ImportError:
    import json
sys.path[0:0] = ['']
from test import client_context, host, port, unittest
from bson import decode, encode
from bson.json_util import loads
from gridfs import GridFSBucket
from pymongo import MongoClient
NUM_ITERATIONS = 100
MAX_ITERATION_TIME = 300
NUM_DOCS = 10000
TEST_PATH = os.environ.get('TEST_PATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join('data')))
OUTPUT_FILE = os.environ.get('OUTPUT_FILE')
result_data: List = []

def tearDownModule():
    if False:
        print('Hello World!')
    output = json.dumps(result_data, indent=4)
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'w') as opf:
            opf.write(output)
    else:
        print(output)

class Timer:

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        self.end = time.monotonic()
        self.interval = self.end - self.start

class PerformanceTest:
    dataset: Any
    data_size: Any
    do_task: Any
    fail: Any

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        client_context.init()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        name = self.__class__.__name__
        median = self.percentile(50)
        bytes_per_sec = self.data_size / median
        print(f'Running {self.__class__.__name__}. MEDIAN={self.percentile(50)}')
        result_data.append({'info': {'test_name': name, 'args': {'threads': 1}}, 'metrics': [{'name': 'bytes_per_sec', 'value': bytes_per_sec}]})

    def before(self):
        if False:
            while True:
                i = 10
        pass

    def after(self):
        if False:
            while True:
                i = 10
        pass

    def percentile(self, percentile):
        if False:
            while True:
                i = 10
        if hasattr(self, 'results'):
            sorted_results = sorted(self.results)
            percentile_index = int(len(sorted_results) * percentile / 100) - 1
            return sorted_results[percentile_index]
        else:
            self.fail('Test execution failed')
            return None

    def runTest(self):
        if False:
            while True:
                i = 10
        results = []
        start = time.monotonic()
        self.max_iterations = NUM_ITERATIONS
        for i in range(NUM_ITERATIONS):
            if time.monotonic() - start > MAX_ITERATION_TIME:
                with warnings.catch_warnings():
                    warnings.simplefilter('default')
                    warnings.warn('Test timed out, completed %s iterations.' % i)
                break
            self.before()
            with Timer() as timer:
                self.do_task()
            self.after()
            results.append(timer.interval)
        self.results = results

class BsonEncodingTest(PerformanceTest):

    def setUp(self):
        if False:
            print('Hello World!')
        with open(os.path.join(TEST_PATH, os.path.join('extended_bson', self.dataset))) as data:
            self.document = loads(data.read())

    def do_task(self):
        if False:
            return 10
        for _ in range(NUM_DOCS):
            encode(self.document)

class BsonDecodingTest(PerformanceTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        with open(os.path.join(TEST_PATH, os.path.join('extended_bson', self.dataset))) as data:
            self.document = encode(json.loads(data.read()))

    def do_task(self):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(NUM_DOCS):
            decode(self.document)

class TestFlatEncoding(BsonEncodingTest, unittest.TestCase):
    dataset = 'flat_bson.json'
    data_size = 75310000

class TestFlatDecoding(BsonDecodingTest, unittest.TestCase):
    dataset = 'flat_bson.json'
    data_size = 75310000

class TestDeepEncoding(BsonEncodingTest, unittest.TestCase):
    dataset = 'deep_bson.json'
    data_size = 19640000

class TestDeepDecoding(BsonDecodingTest, unittest.TestCase):
    dataset = 'deep_bson.json'
    data_size = 19640000

class TestFullEncoding(BsonEncodingTest, unittest.TestCase):
    dataset = 'full_bson.json'
    data_size = 57340000

class TestFullDecoding(BsonDecodingTest, unittest.TestCase):
    dataset = 'full_bson.json'
    data_size = 57340000

class TestRunCommand(PerformanceTest, unittest.TestCase):
    data_size = 160000

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.client = client_context.client
        self.client.drop_database('perftest')

    def do_task(self):
        if False:
            while True:
                i = 10
        command = self.client.perftest.command
        for _ in range(NUM_DOCS):
            command('ping')

class TestDocument(PerformanceTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with open(os.path.join(TEST_PATH, os.path.join('single_and_multi_document', self.dataset))) as data:
            self.document = json.loads(data.read())
        self.client = client_context.client
        self.client.drop_database('perftest')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        self.client.drop_database('perftest')

    def before(self):
        if False:
            while True:
                i = 10
        self.corpus = self.client.perftest.create_collection('corpus')

    def after(self):
        if False:
            return 10
        self.client.perftest.drop_collection('corpus')

class TestFindOneByID(TestDocument, unittest.TestCase):
    data_size = 16220000

    def setUp(self):
        if False:
            print('Hello World!')
        self.dataset = 'tweet.json'
        super().setUp()
        documents = [self.document.copy() for _ in range(NUM_DOCS)]
        self.corpus = self.client.perftest.corpus
        result = self.corpus.insert_many(documents)
        self.inserted_ids = result.inserted_ids

    def do_task(self):
        if False:
            while True:
                i = 10
        find_one = self.corpus.find_one
        for _id in self.inserted_ids:
            find_one({'_id': _id})

    def before(self):
        if False:
            return 10
        pass

    def after(self):
        if False:
            i = 10
            return i + 15
        pass

class TestSmallDocInsertOne(TestDocument, unittest.TestCase):
    data_size = 2750000

    def setUp(self):
        if False:
            return 10
        self.dataset = 'small_doc.json'
        super().setUp()
        self.documents = [self.document.copy() for _ in range(NUM_DOCS)]

    def do_task(self):
        if False:
            return 10
        insert_one = self.corpus.insert_one
        for doc in self.documents:
            insert_one(doc)

class TestLargeDocInsertOne(TestDocument, unittest.TestCase):
    data_size = 27310890

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.dataset = 'large_doc.json'
        super().setUp()
        self.documents = [self.document.copy() for _ in range(10)]

    def do_task(self):
        if False:
            return 10
        insert_one = self.corpus.insert_one
        for doc in self.documents:
            insert_one(doc)

class TestFindManyAndEmptyCursor(TestDocument, unittest.TestCase):
    data_size = 16220000

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.dataset = 'tweet.json'
        super().setUp()
        for _ in range(10):
            self.client.perftest.command('insert', 'corpus', documents=[self.document] * 1000)
        self.corpus = self.client.perftest.corpus

    def do_task(self):
        if False:
            return 10
        list(self.corpus.find())

    def before(self):
        if False:
            while True:
                i = 10
        pass

    def after(self):
        if False:
            i = 10
            return i + 15
        pass

class TestSmallDocBulkInsert(TestDocument, unittest.TestCase):
    data_size = 2750000

    def setUp(self):
        if False:
            print('Hello World!')
        self.dataset = 'small_doc.json'
        super().setUp()
        self.documents = [self.document.copy() for _ in range(NUM_DOCS)]

    def before(self):
        if False:
            i = 10
            return i + 15
        self.corpus = self.client.perftest.create_collection('corpus')

    def do_task(self):
        if False:
            return 10
        self.corpus.insert_many(self.documents, ordered=True)

class TestLargeDocBulkInsert(TestDocument, unittest.TestCase):
    data_size = 27310890

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.dataset = 'large_doc.json'
        super().setUp()
        self.documents = [self.document.copy() for _ in range(10)]

    def before(self):
        if False:
            i = 10
            return i + 15
        self.corpus = self.client.perftest.create_collection('corpus')

    def do_task(self):
        if False:
            print('Hello World!')
        self.corpus.insert_many(self.documents, ordered=True)

class TestGridFsUpload(PerformanceTest, unittest.TestCase):
    data_size = 52428800

    def setUp(self):
        if False:
            while True:
                i = 10
        self.client = client_context.client
        self.client.drop_database('perftest')
        gridfs_path = os.path.join(TEST_PATH, os.path.join('single_and_multi_document', 'gridfs_large.bin'))
        with open(gridfs_path, 'rb') as data:
            self.document = data.read()
        self.bucket = GridFSBucket(self.client.perftest)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        self.client.drop_database('perftest')

    def before(self):
        if False:
            for i in range(10):
                print('nop')
        self.bucket.upload_from_stream('init', b'x')

    def do_task(self):
        if False:
            for i in range(10):
                print('nop')
        self.bucket.upload_from_stream('gridfstest', self.document)

class TestGridFsDownload(PerformanceTest, unittest.TestCase):
    data_size = 52428800

    def setUp(self):
        if False:
            while True:
                i = 10
        self.client = client_context.client
        self.client.drop_database('perftest')
        gridfs_path = os.path.join(TEST_PATH, os.path.join('single_and_multi_document', 'gridfs_large.bin'))
        self.bucket = GridFSBucket(self.client.perftest)
        with open(gridfs_path, 'rb') as gfile:
            self.uploaded_id = self.bucket.upload_from_stream('gridfstest', gfile)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        self.client.drop_database('perftest')

    def do_task(self):
        if False:
            for i in range(10):
                print('nop')
        self.bucket.open_download_stream(self.uploaded_id).read()
proc_client = None

def proc_init(*dummy):
    if False:
        for i in range(10):
            print('nop')
    global proc_client
    proc_client = MongoClient(host, port)

def mp_map(map_func, files):
    if False:
        return 10
    pool = mp.Pool(initializer=proc_init)
    pool.map(map_func, files)
    pool.close()

def insert_json_file(filename):
    if False:
        return 10
    assert proc_client is not None
    with open(filename) as data:
        coll = proc_client.perftest.corpus
        coll.insert_many([json.loads(line) for line in data])

def insert_json_file_with_file_id(filename):
    if False:
        print('Hello World!')
    documents = []
    with open(filename) as data:
        for line in data:
            doc = json.loads(line)
            doc['file'] = filename
            documents.append(doc)
    assert proc_client is not None
    coll = proc_client.perftest.corpus
    coll.insert_many(documents)

def read_json_file(filename):
    if False:
        return 10
    assert proc_client is not None
    coll = proc_client.perftest.corpus
    temp = tempfile.TemporaryFile(mode='w')
    try:
        temp.writelines([json.dumps(doc) + '\n' for doc in coll.find({'file': filename}, {'_id': False})])
    finally:
        temp.close()

def insert_gridfs_file(filename):
    if False:
        i = 10
        return i + 15
    assert proc_client is not None
    bucket = GridFSBucket(proc_client.perftest)
    with open(filename, 'rb') as gfile:
        bucket.upload_from_stream(filename, gfile)

def read_gridfs_file(filename):
    if False:
        print('Hello World!')
    assert proc_client is not None
    bucket = GridFSBucket(proc_client.perftest)
    temp = tempfile.TemporaryFile()
    try:
        bucket.download_to_stream_by_name(filename, temp)
    finally:
        temp.close()

class TestJsonMultiImport(PerformanceTest, unittest.TestCase):
    data_size = 565000000

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.client = client_context.client
        self.client.drop_database('perftest')

    def before(self):
        if False:
            print('Hello World!')
        self.client.perftest.command({'create': 'corpus'})
        self.corpus = self.client.perftest.corpus
        ldjson_path = os.path.join(TEST_PATH, os.path.join('parallel', 'ldjson_multi'))
        self.files = [os.path.join(ldjson_path, s) for s in os.listdir(ldjson_path)]

    def do_task(self):
        if False:
            return 10
        mp_map(insert_json_file, self.files)

    def after(self):
        if False:
            while True:
                i = 10
        self.client.perftest.drop_collection('corpus')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        self.client.drop_database('perftest')

class TestJsonMultiExport(PerformanceTest, unittest.TestCase):
    data_size = 565000000

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.client = client_context.client
        self.client.drop_database('perftest')
        self.client.perfest.corpus.create_index('file')
        ldjson_path = os.path.join(TEST_PATH, os.path.join('parallel', 'ldjson_multi'))
        self.files = [os.path.join(ldjson_path, s) for s in os.listdir(ldjson_path)]
        mp_map(insert_json_file_with_file_id, self.files)

    def do_task(self):
        if False:
            return 10
        mp_map(read_json_file, self.files)

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        self.client.drop_database('perftest')

class TestGridFsMultiFileUpload(PerformanceTest, unittest.TestCase):
    data_size = 262144000

    def setUp(self):
        if False:
            while True:
                i = 10
        self.client = client_context.client
        self.client.drop_database('perftest')

    def before(self):
        if False:
            i = 10
            return i + 15
        self.client.perftest.drop_collection('fs.files')
        self.client.perftest.drop_collection('fs.chunks')
        self.bucket = GridFSBucket(self.client.perftest)
        gridfs_path = os.path.join(TEST_PATH, os.path.join('parallel', 'gridfs_multi'))
        self.files = [os.path.join(gridfs_path, s) for s in os.listdir(gridfs_path)]

    def do_task(self):
        if False:
            for i in range(10):
                print('nop')
        mp_map(insert_gridfs_file, self.files)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        self.client.drop_database('perftest')

class TestGridFsMultiFileDownload(PerformanceTest, unittest.TestCase):
    data_size = 262144000

    def setUp(self):
        if False:
            while True:
                i = 10
        self.client = client_context.client
        self.client.drop_database('perftest')
        bucket = GridFSBucket(self.client.perftest)
        gridfs_path = os.path.join(TEST_PATH, os.path.join('parallel', 'gridfs_multi'))
        self.files = [os.path.join(gridfs_path, s) for s in os.listdir(gridfs_path)]
        for fname in self.files:
            with open(fname, 'rb') as gfile:
                bucket.upload_from_stream(fname, gfile)

    def do_task(self):
        if False:
            for i in range(10):
                print('nop')
        mp_map(read_gridfs_file, self.files)

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        self.client.drop_database('perftest')
if __name__ == '__main__':
    unittest.main()
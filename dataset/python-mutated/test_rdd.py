from datetime import datetime, timedelta
import hashlib
import os
import random
import tempfile
import time
import unittest
from glob import glob
from py4j.protocol import Py4JJavaError
from pyspark import shuffle, RDD
from pyspark.resource import ExecutorResourceRequests, ResourceProfileBuilder, TaskResourceRequests
from pyspark.serializers import CloudPickleSerializer, BatchedSerializer, CPickleSerializer, MarshalSerializer, UTF8Deserializer, NoOpSerializer
from pyspark.sql import SparkSession
from pyspark.testing.utils import ReusedPySparkTestCase, SPARK_HOME, QuietTest, have_numpy
from pyspark.testing.sqlutils import have_pandas
global_func = lambda : 'Hi'

class RDDTests(ReusedPySparkTestCase):

    def test_range(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.sc.range(1, 1).count(), 0)
        self.assertEqual(self.sc.range(1, 0, -1).count(), 1)
        self.assertEqual(self.sc.range(0, 1 << 40, 1 << 39).count(), 2)

    def test_id(self):
        if False:
            i = 10
            return i + 15
        rdd = self.sc.parallelize(range(10))
        id = rdd.id()
        self.assertEqual(id, rdd.id())
        rdd2 = rdd.map(str).filter(bool)
        id2 = rdd2.id()
        self.assertEqual(id + 1, id2)
        self.assertEqual(id2, rdd2.id())

    def test_empty_rdd(self):
        if False:
            i = 10
            return i + 15
        rdd = self.sc.emptyRDD()
        self.assertTrue(rdd.isEmpty())

    def test_sum(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, self.sc.emptyRDD().sum())
        self.assertEqual(6, self.sc.parallelize([1, 2, 3]).sum())

    def test_to_localiterator(self):
        if False:
            i = 10
            return i + 15
        rdd = self.sc.parallelize([1, 2, 3])
        it = rdd.toLocalIterator()
        self.assertEqual([1, 2, 3], sorted(it))
        rdd2 = rdd.repartition(1000)
        it2 = rdd2.toLocalIterator()
        self.assertEqual([1, 2, 3], sorted(it2))

    def test_to_localiterator_prefetch(self):
        if False:
            while True:
                i = 10
        rdd = self.sc.parallelize(range(2), 2)
        times1 = rdd.map(lambda x: datetime.now())
        times2 = rdd.map(lambda x: datetime.now())
        times_iter_prefetch = times1.toLocalIterator(prefetchPartitions=True)
        times_iter = times2.toLocalIterator(prefetchPartitions=False)
        times_prefetch_head = next(times_iter_prefetch)
        times_head = next(times_iter)
        time.sleep(2)
        times_next = next(times_iter)
        times_prefetch_next = next(times_iter_prefetch)
        self.assertTrue(times_next - times_head >= timedelta(seconds=2))
        self.assertTrue(times_prefetch_next - times_prefetch_head < timedelta(seconds=1))

    def test_save_as_textfile_with_unicode(self):
        if False:
            print('Hello World!')
        x = '¡Hola, mundo!'
        data = self.sc.parallelize([x])
        tempFile = tempfile.NamedTemporaryFile(delete=True)
        tempFile.close()
        data.saveAsTextFile(tempFile.name)
        raw_contents = b''.join((open(p, 'rb').read() for p in glob(tempFile.name + '/part-0000*')))
        self.assertEqual(x, raw_contents.strip().decode('utf-8'))

    def test_save_as_textfile_with_utf8(self):
        if False:
            i = 10
            return i + 15
        x = '¡Hola, mundo!'
        data = self.sc.parallelize([x.encode('utf-8')])
        tempFile = tempfile.NamedTemporaryFile(delete=True)
        tempFile.close()
        data.saveAsTextFile(tempFile.name)
        raw_contents = b''.join((open(p, 'rb').read() for p in glob(tempFile.name + '/part-0000*')))
        self.assertEqual(x, raw_contents.strip().decode('utf8'))

    def test_transforming_cartesian_result(self):
        if False:
            for i in range(10):
                print('nop')
        rdd1 = self.sc.parallelize([1, 2])
        rdd2 = self.sc.parallelize([3, 4])
        cart = rdd1.cartesian(rdd2)
        cart.map(lambda x_y3: x_y3[0] + x_y3[1]).collect()

    def test_transforming_pickle_file(self):
        if False:
            i = 10
            return i + 15
        data = self.sc.parallelize(['Hello', 'World!'])
        tempFile = tempfile.NamedTemporaryFile(delete=True)
        tempFile.close()
        data.saveAsPickleFile(tempFile.name)
        pickled_file = self.sc.pickleFile(tempFile.name)
        pickled_file.map(lambda x: x).collect()

    def test_cartesian_on_textfile(self):
        if False:
            while True:
                i = 10
        path = os.path.join(SPARK_HOME, 'python/test_support/hello/hello.txt')
        a = self.sc.textFile(path)
        result = a.cartesian(a).collect()
        (x, y) = result[0]
        self.assertEqual('Hello World!', x.strip())
        self.assertEqual('Hello World!', y.strip())

    def test_cartesian_chaining(self):
        if False:
            i = 10
            return i + 15
        rdd = self.sc.parallelize(range(10), 2)
        self.assertSetEqual(set(rdd.cartesian(rdd).cartesian(rdd).collect()), set([((x, y), z) for x in range(10) for y in range(10) for z in range(10)]))
        self.assertSetEqual(set(rdd.cartesian(rdd.cartesian(rdd)).collect()), set([(x, (y, z)) for x in range(10) for y in range(10) for z in range(10)]))
        self.assertSetEqual(set(rdd.cartesian(rdd.zip(rdd)).collect()), set([(x, (y, y)) for x in range(10) for y in range(10)]))

    def test_zip_chaining(self):
        if False:
            i = 10
            return i + 15
        rdd = self.sc.parallelize('abc', 2)
        self.assertSetEqual(set(rdd.zip(rdd).zip(rdd).collect()), set([((x, x), x) for x in 'abc']))
        self.assertSetEqual(set(rdd.zip(rdd.zip(rdd)).collect()), set([(x, (x, x)) for x in 'abc']))

    def test_union_pair_rdd(self):
        if False:
            for i in range(10):
                print('nop')
        rdd = self.sc.parallelize([1, 2])
        pair_rdd = rdd.zip(rdd)
        unionRDD = self.sc.union([pair_rdd, pair_rdd])
        self.assertEqual(set(unionRDD.collect()), set([(1, 1), (2, 2), (1, 1), (2, 2)]))
        self.assertEqual(unionRDD.count(), 4)

    def test_deleting_input_files(self):
        if False:
            print('Hello World!')
        tempFile = tempfile.NamedTemporaryFile(delete=False)
        tempFile.write(b'Hello World!')
        tempFile.close()
        data = self.sc.textFile(tempFile.name)
        filtered_data = data.filter(lambda x: True)
        self.assertEqual(1, filtered_data.count())
        os.unlink(tempFile.name)
        with QuietTest(self.sc):
            self.assertRaises(Exception, lambda : filtered_data.count())

    def test_sampling_default_seed(self):
        if False:
            return 10
        data = self.sc.parallelize(range(1000), 1)
        subset = data.takeSample(False, 10)
        self.assertEqual(len(subset), 10)

    def test_aggregate_mutable_zero_value(self):
        if False:
            while True:
                i = 10
        from collections import defaultdict
        data1 = self.sc.range(10, numSlices=1)
        data2 = self.sc.range(10, numSlices=2)

        def seqOp(x, y):
            if False:
                print('Hello World!')
            x[y] += 1
            return x

        def comboOp(x, y):
            if False:
                print('Hello World!')
            for (key, val) in y.items():
                x[key] += val
            return x
        counts1 = data1.aggregate(defaultdict(int), seqOp, comboOp)
        counts2 = data2.aggregate(defaultdict(int), seqOp, comboOp)
        counts3 = data1.treeAggregate(defaultdict(int), seqOp, comboOp, 2)
        counts4 = data2.treeAggregate(defaultdict(int), seqOp, comboOp, 2)
        ground_truth = defaultdict(int, dict(((i, 1) for i in range(10))))
        self.assertEqual(counts1, ground_truth)
        self.assertEqual(counts2, ground_truth)
        self.assertEqual(counts3, ground_truth)
        self.assertEqual(counts4, ground_truth)

    def test_aggregate_by_key_mutable_zero_value(self):
        if False:
            print('Hello World!')
        tuples = list(zip(list(range(10)) * 2, [1] * 20))
        data1 = self.sc.parallelize(tuples, 1)
        data2 = self.sc.parallelize(tuples, 2)

        def seqOp(x, y):
            if False:
                print('Hello World!')
            x.append(y)
            return x

        def comboOp(x, y):
            if False:
                while True:
                    i = 10
            x.extend(y)
            return x
        values1 = data1.aggregateByKey([], seqOp, comboOp).collect()
        values2 = data2.aggregateByKey([], seqOp, comboOp).collect()
        values1.sort()
        values2.sort()
        ground_truth = [(i, [1] * 2) for i in range(10)]
        self.assertEqual(values1, ground_truth)
        self.assertEqual(values2, ground_truth)

    def test_fold_mutable_zero_value(self):
        if False:
            i = 10
            return i + 15
        from collections import defaultdict
        counts1 = defaultdict(int, dict(((i, 1) for i in range(10))))
        counts2 = defaultdict(int, dict(((i, 1) for i in range(3, 8))))
        counts3 = defaultdict(int, dict(((i, 1) for i in range(4, 7))))
        counts4 = defaultdict(int, dict(((i, 1) for i in range(5, 6))))
        all_counts = [counts1, counts2, counts3, counts4]
        data1 = self.sc.parallelize(all_counts, 1)
        data2 = self.sc.parallelize(all_counts, 2)

        def comboOp(x, y):
            if False:
                print('Hello World!')
            for (key, val) in y.items():
                x[key] += val
            return x
        fold1 = data1.fold(defaultdict(int), comboOp)
        fold2 = data2.fold(defaultdict(int), comboOp)
        ground_truth = defaultdict(int)
        for counts in all_counts:
            for (key, val) in counts.items():
                ground_truth[key] += val
        self.assertEqual(fold1, ground_truth)
        self.assertEqual(fold2, ground_truth)

    def test_fold_by_key_mutable_zero_value(self):
        if False:
            for i in range(10):
                print('nop')
        tuples = [(i, range(i)) for i in range(10)] * 2
        data1 = self.sc.parallelize(tuples, 1)
        data2 = self.sc.parallelize(tuples, 2)

        def comboOp(x, y):
            if False:
                for i in range(10):
                    print('nop')
            x.extend(y)
            return x
        values1 = data1.foldByKey([], comboOp).collect()
        values2 = data2.foldByKey([], comboOp).collect()
        values1.sort()
        values2.sort()
        ground_truth = [(i, list(range(i)) * 2) for i in range(10)]
        self.assertEqual(values1, ground_truth)
        self.assertEqual(values2, ground_truth)

    def test_aggregate_by_key(self):
        if False:
            while True:
                i = 10
        data = self.sc.parallelize([(1, 1), (1, 1), (3, 2), (5, 1), (5, 3)], 2)

        def seqOp(x, y):
            if False:
                i = 10
                return i + 15
            x.add(y)
            return x

        def combOp(x, y):
            if False:
                print('Hello World!')
            x |= y
            return x
        sets = dict(data.aggregateByKey(set(), seqOp, combOp).collect())
        self.assertEqual(3, len(sets))
        self.assertEqual(set([1]), sets[1])
        self.assertEqual(set([2]), sets[3])
        self.assertEqual(set([1, 3]), sets[5])

    def test_itemgetter(self):
        if False:
            print('Hello World!')
        rdd = self.sc.parallelize([range(10)])
        from operator import itemgetter
        self.assertEqual([1], rdd.map(itemgetter(1)).collect())
        self.assertEqual([(2, 3)], rdd.map(itemgetter(2, 3)).collect())

    def test_namedtuple_in_rdd(self):
        if False:
            while True:
                i = 10
        from collections import namedtuple
        Person = namedtuple('Person', 'id firstName lastName')
        jon = Person(1, 'Jon', 'Doe')
        jane = Person(2, 'Jane', 'Doe')
        theDoes = self.sc.parallelize([jon, jane])
        self.assertEqual([jon, jane], theDoes.collect())

    def test_large_broadcast(self):
        if False:
            return 10
        N = 10000
        data = [[float(i) for i in range(300)] for i in range(N)]
        bdata = self.sc.broadcast(data)
        m = self.sc.parallelize(range(1), 1).map(lambda x: len(bdata.value)).sum()
        self.assertEqual(N, m)

    def test_unpersist(self):
        if False:
            while True:
                i = 10
        N = 1000
        data = [[float(i) for i in range(300)] for i in range(N)]
        bdata = self.sc.broadcast(data)
        bdata.unpersist()
        m = self.sc.parallelize(range(1), 1).map(lambda x: len(bdata.value)).sum()
        self.assertEqual(N, m)
        bdata.destroy(blocking=True)
        try:
            self.sc.parallelize(range(1), 1).map(lambda x: len(bdata.value)).sum()
        except Exception:
            pass
        else:
            raise AssertionError('job should fail after destroy the broadcast')

    def test_multiple_broadcasts(self):
        if False:
            i = 10
            return i + 15
        N = 1 << 21
        b1 = self.sc.broadcast(set(range(N)))
        r = list(range(1 << 15))
        random.shuffle(r)
        s = str(r).encode()
        checksum = hashlib.md5(s).hexdigest()
        b2 = self.sc.broadcast(s)
        r = list(set(self.sc.parallelize(range(10), 10).map(lambda x: (len(b1.value), hashlib.md5(b2.value).hexdigest())).collect()))
        self.assertEqual(1, len(r))
        (size, csum) = r[0]
        self.assertEqual(N, size)
        self.assertEqual(checksum, csum)
        random.shuffle(r)
        s = str(r).encode()
        checksum = hashlib.md5(s).hexdigest()
        b2 = self.sc.broadcast(s)
        r = list(set(self.sc.parallelize(range(10), 10).map(lambda x: (len(b1.value), hashlib.md5(b2.value).hexdigest())).collect()))
        self.assertEqual(1, len(r))
        (size, csum) = r[0]
        self.assertEqual(N, size)
        self.assertEqual(checksum, csum)

    def test_multithread_broadcast_pickle(self):
        if False:
            while True:
                i = 10
        import threading
        b1 = self.sc.broadcast(list(range(3)))
        b2 = self.sc.broadcast(list(range(3)))

        def f1():
            if False:
                print('Hello World!')
            return b1.value

        def f2():
            if False:
                return 10
            return b2.value
        funcs_num_pickled = {f1: None, f2: None}

        def do_pickle(f, sc):
            if False:
                return 10
            command = (f, None, sc.serializer, sc.serializer)
            ser = CloudPickleSerializer()
            ser.dumps(command)

        def process_vars(sc):
            if False:
                return 10
            broadcast_vars = list(sc._pickled_broadcast_vars)
            num_pickled = len(broadcast_vars)
            sc._pickled_broadcast_vars.clear()
            return num_pickled

        def run(f, sc):
            if False:
                for i in range(10):
                    print('nop')
            do_pickle(f, sc)
            funcs_num_pickled[f] = process_vars(sc)
        do_pickle(f1, self.sc)
        t = threading.Thread(target=run, args=(f2, self.sc))
        t.start()
        t.join()
        funcs_num_pickled[f1] = process_vars(self.sc)
        self.assertEqual(funcs_num_pickled[f1], 1)
        self.assertEqual(funcs_num_pickled[f2], 1)
        self.assertEqual(len(list(self.sc._pickled_broadcast_vars)), 0)

    def test_large_closure(self):
        if False:
            while True:
                i = 10
        N = 200000
        data = [float(i) for i in range(N)]
        rdd = self.sc.parallelize(range(1), 1).map(lambda x: len(data))
        self.assertEqual(N, rdd.first())
        self.assertEqual(1, rdd.map(lambda x: (x, 1)).groupByKey().count())

    def test_zip_with_different_serializers(self):
        if False:
            print('Hello World!')
        a = self.sc.parallelize(range(5))
        b = self.sc.parallelize(range(100, 105))
        self.assertEqual(a.zip(b).collect(), [(0, 100), (1, 101), (2, 102), (3, 103), (4, 104)])
        a = a._reserialize(BatchedSerializer(CPickleSerializer(), 2))
        b = b._reserialize(MarshalSerializer())
        self.assertEqual(a.zip(b).collect(), [(0, 100), (1, 101), (2, 102), (3, 103), (4, 104)])
        path = os.path.join(SPARK_HOME, 'python/test_support/hello/hello.txt')
        t = self.sc.textFile(path)
        cnt = t.count()
        self.assertEqual(cnt, t.zip(t).count())
        rdd = t.map(str)
        self.assertEqual(cnt, t.zip(rdd).count())
        self.assertEqual(cnt, t.zip(rdd).count())

    def test_zip_with_different_object_sizes(self):
        if False:
            while True:
                i = 10
        a = self.sc.parallelize(range(10000)).map(lambda i: '*' * i)
        b = self.sc.parallelize(range(10000, 20000)).map(lambda i: '*' * i)
        self.assertEqual(10000, a.zip(b).count())

    def test_zip_with_different_number_of_items(self):
        if False:
            print('Hello World!')
        a = self.sc.parallelize(range(5), 2)
        b = self.sc.parallelize(range(100, 106), 3)
        self.assertRaises(ValueError, lambda : a.zip(b))
        with QuietTest(self.sc):
            b = self.sc.parallelize(range(100, 104), 2)
            self.assertRaises(Exception, lambda : a.zip(b).count())
            b = self.sc.parallelize(range(100, 106), 2)
            self.assertRaises(Exception, lambda : a.zip(b).count())
            a = self.sc.parallelize([2, 3], 2).flatMap(range)
            b = self.sc.parallelize([3, 2], 2).flatMap(range)
            self.assertEqual(a.count(), b.count())
            self.assertRaises(Exception, lambda : a.zip(b).count())

    def test_count_approx_distinct(self):
        if False:
            print('Hello World!')
        rdd = self.sc.parallelize(range(1000))
        self.assertTrue(950 < rdd.countApproxDistinct(0.03) < 1050)
        self.assertTrue(950 < rdd.map(float).countApproxDistinct(0.03) < 1050)
        self.assertTrue(950 < rdd.map(str).countApproxDistinct(0.03) < 1050)
        self.assertTrue(950 < rdd.map(lambda x: (x, -x)).countApproxDistinct(0.03) < 1050)
        rdd = self.sc.parallelize([i % 20 for i in range(1000)], 7)
        self.assertTrue(18 < rdd.countApproxDistinct() < 22)
        self.assertTrue(18 < rdd.map(float).countApproxDistinct() < 22)
        self.assertTrue(18 < rdd.map(str).countApproxDistinct() < 22)
        self.assertTrue(18 < rdd.map(lambda x: (x, -x)).countApproxDistinct() < 22)
        self.assertRaises(ValueError, lambda : rdd.countApproxDistinct(1e-08))

    def test_histogram(self):
        if False:
            for i in range(10):
                print('nop')
        rdd = self.sc.parallelize([])
        self.assertEqual([0], rdd.histogram([0, 10])[1])
        self.assertEqual([0, 0], rdd.histogram([0, 4, 10])[1])
        self.assertRaises(ValueError, lambda : rdd.histogram(1))
        rdd = self.sc.parallelize([10.01, -0.01])
        self.assertEqual([0], rdd.histogram([0, 10])[1])
        self.assertEqual([0, 0], rdd.histogram((0, 4, 10))[1])
        rdd = self.sc.parallelize(range(1, 5))
        self.assertEqual([4], rdd.histogram([0, 10])[1])
        self.assertEqual([3, 1], rdd.histogram([0, 4, 10])[1])
        self.assertEqual([4], rdd.histogram([1, 4])[1])
        rdd = self.sc.parallelize([10.01, -0.01])
        self.assertEqual([0, 0], rdd.histogram([0, 5, 10])[1])
        rdd = self.sc.parallelize([10.01, -0.01])
        self.assertEqual([0, 0], rdd.histogram([0, 4, 10])[1])
        rdd = self.sc.parallelize([1, 2, 3, 5, 6])
        self.assertEqual([3, 2], rdd.histogram([0, 5, 10])[1])
        rdd = self.sc.parallelize([1, 2, 3, 5, 6, None, float('nan')])
        self.assertEqual([3, 2], rdd.histogram([0, 5, 10])[1])
        rdd = self.sc.parallelize([1, 2, 3, 5, 6])
        self.assertEqual([3, 2], rdd.histogram([0, 5, 11])[1])
        rdd = self.sc.parallelize([-0.01, 0.0, 1, 2, 3, 5, 6, 11.0, 11.01])
        self.assertEqual([4, 3], rdd.histogram([0, 5, 11])[1])
        rdd = self.sc.parallelize([-0.01, 0.0, 1, 2, 3, 5, 6, 11.01, 12.0, 199.0, 200.0, 200.1])
        self.assertEqual([4, 2, 1, 3], rdd.histogram([0.0, 5.0, 11.0, 12.0, 200.0])[1])
        rdd = self.sc.parallelize([-0.01, 0.0, 1, 2, 3, 5, 6, 11.01, 12.0, 199.0, 200.0, 200.1, None, float('nan')])
        self.assertEqual([4, 2, 1, 3], rdd.histogram([0.0, 5.0, 11.0, 12.0, 200.0])[1])
        rdd = self.sc.parallelize([10.01, -0.01, float('nan'), float('inf')])
        self.assertEqual([1, 2], rdd.histogram([float('-inf'), 0, float('inf')])[1])
        self.assertRaises(ValueError, lambda : rdd.histogram([]))
        self.assertRaises(ValueError, lambda : rdd.histogram([1]))
        self.assertRaises(ValueError, lambda : rdd.histogram(0))
        self.assertRaises(TypeError, lambda : rdd.histogram({}))
        rdd = self.sc.parallelize(range(1, 5))
        self.assertEqual(([1, 4], [4]), rdd.histogram(1))
        rdd = self.sc.parallelize([1])
        self.assertEqual(([1, 1], [1]), rdd.histogram(1))
        rdd = self.sc.parallelize([1] * 4)
        self.assertEqual(([1, 1], [4]), rdd.histogram(1))
        rdd = self.sc.parallelize(range(1, 5))
        self.assertEqual(([1, 2.5, 4], [2, 2]), rdd.histogram(2))
        rdd = self.sc.parallelize([1, 2])
        buckets = [1 + 0.2 * i for i in range(6)]
        hist = [1, 0, 0, 0, 1]
        self.assertEqual((buckets, hist), rdd.histogram(5))
        rdd = self.sc.parallelize([1, float('inf')])
        self.assertRaises(ValueError, lambda : rdd.histogram(2))
        rdd = self.sc.parallelize([float('nan')])
        self.assertRaises(ValueError, lambda : rdd.histogram(2))
        rdd = self.sc.parallelize(['ab', 'ac', 'b', 'bd', 'ef'], 2)
        self.assertEqual([2, 2], rdd.histogram(['a', 'b', 'c'])[1])
        self.assertEqual((['ab', 'ef'], [5]), rdd.histogram(1))
        self.assertRaises(TypeError, lambda : rdd.histogram(2))

    def test_repartitionAndSortWithinPartitions_asc(self):
        if False:
            while True:
                i = 10
        rdd = self.sc.parallelize([(0, 5), (3, 8), (2, 6), (0, 8), (3, 8), (1, 3)], 2)
        repartitioned = rdd.repartitionAndSortWithinPartitions(2, lambda key: key % 2, True)
        partitions = repartitioned.glom().collect()
        self.assertEqual(partitions[0], [(0, 5), (0, 8), (2, 6)])
        self.assertEqual(partitions[1], [(1, 3), (3, 8), (3, 8)])

    def test_repartitionAndSortWithinPartitions_desc(self):
        if False:
            while True:
                i = 10
        rdd = self.sc.parallelize([(0, 5), (3, 8), (2, 6), (0, 8), (3, 8), (1, 3)], 2)
        repartitioned = rdd.repartitionAndSortWithinPartitions(2, lambda key: key % 2, False)
        partitions = repartitioned.glom().collect()
        self.assertEqual(partitions[0], [(2, 6), (0, 5), (0, 8)])
        self.assertEqual(partitions[1], [(3, 8), (3, 8), (1, 3)])

    def test_repartition_no_skewed(self):
        if False:
            return 10
        num_partitions = 20
        a = self.sc.parallelize(range(int(1000)), 2)
        xs = a.repartition(num_partitions).glom().map(len).collect()
        zeros = len([x for x in xs if x == 0])
        self.assertTrue(zeros == 0)
        xs = a.coalesce(num_partitions, True).glom().map(len).collect()
        zeros = len([x for x in xs if x == 0])
        self.assertTrue(zeros == 0)

    def test_repartition_on_textfile(self):
        if False:
            while True:
                i = 10
        path = os.path.join(SPARK_HOME, 'python/test_support/hello/hello.txt')
        rdd = self.sc.textFile(path)
        result = rdd.repartition(1).collect()
        self.assertEqual('Hello World!', result[0])

    def test_distinct(self):
        if False:
            print('Hello World!')
        rdd = self.sc.parallelize((1, 2, 3) * 10, 10)
        self.assertEqual(rdd.getNumPartitions(), 10)
        self.assertEqual(rdd.distinct().count(), 3)
        result = rdd.distinct(5)
        self.assertEqual(result.getNumPartitions(), 5)
        self.assertEqual(result.count(), 3)

    def test_external_group_by_key(self):
        if False:
            for i in range(10):
                print('nop')
        self.sc._conf.set('spark.python.worker.memory', '1m')
        N = 2000001
        kv = self.sc.parallelize(range(N)).map(lambda x: (x % 3, x))
        gkv = kv.groupByKey().cache()
        self.assertEqual(3, gkv.count())
        filtered = gkv.filter(lambda kv: kv[0] == 1)
        self.assertEqual(1, filtered.count())
        self.assertEqual([(1, N // 3)], filtered.mapValues(len).collect())
        self.assertEqual([(N // 3, N // 3)], filtered.values().map(lambda x: (len(x), len(list(x)))).collect())
        result = filtered.collect()[0][1]
        self.assertEqual(N // 3, len(result))
        self.assertTrue(isinstance(result.data, shuffle.ExternalListOfList))

    def test_sort_on_empty_rdd(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual([], self.sc.parallelize(zip([], [])).sortByKey().collect())

    def test_sample(self):
        if False:
            print('Hello World!')
        rdd = self.sc.parallelize(range(0, 100), 4)
        wo = rdd.sample(False, 0.1, 2).collect()
        wo_dup = rdd.sample(False, 0.1, 2).collect()
        self.assertSetEqual(set(wo), set(wo_dup))
        wr = rdd.sample(True, 0.2, 5).collect()
        wr_dup = rdd.sample(True, 0.2, 5).collect()
        self.assertSetEqual(set(wr), set(wr_dup))
        wo_s10 = rdd.sample(False, 0.3, 10).collect()
        wo_s20 = rdd.sample(False, 0.3, 20).collect()
        self.assertNotEqual(set(wo_s10), set(wo_s20))
        wr_s11 = rdd.sample(True, 0.4, 11).collect()
        wr_s21 = rdd.sample(True, 0.4, 21).collect()
        self.assertNotEqual(set(wr_s11), set(wr_s21))

    def test_null_in_rdd(self):
        if False:
            while True:
                i = 10
        jrdd = self.sc._jvm.PythonUtils.generateRDDWithNull(self.sc._jsc)
        rdd = RDD(jrdd, self.sc, UTF8Deserializer())
        self.assertEqual(['a', None, 'b'], rdd.collect())
        rdd = RDD(jrdd, self.sc, NoOpSerializer())
        self.assertEqual([b'a', None, b'b'], rdd.collect())

    def test_multiple_python_java_RDD_conversions(self):
        if False:
            return 10
        data = [('1', {'director': 'David Lean'}), ('2', {'director': 'Andrew Dominik'})]
        data_rdd = self.sc.parallelize(data)
        data_java_rdd = data_rdd._to_java_object_rdd()
        data_python_rdd = self.sc._jvm.SerDeUtil.javaToPython(data_java_rdd)
        converted_rdd = RDD(data_python_rdd, self.sc)
        self.assertEqual(2, converted_rdd.count())
        data_java_rdd = converted_rdd._to_java_object_rdd()
        data_python_rdd = self.sc._jvm.SerDeUtil.javaToPython(data_java_rdd)
        converted_rdd = RDD(data_python_rdd, self.sc)
        self.assertEqual(2, converted_rdd.count())

    def test_take_on_jrdd(self):
        if False:
            while True:
                i = 10
        rdd = self.sc.parallelize(range(1 << 20)).map(lambda x: str(x))
        rdd._jrdd.first()

    @unittest.skipIf(not have_numpy or not have_pandas, 'NumPy or Pandas not installed')
    def test_take_on_jrdd_with_large_rows_should_not_cause_deadlock(self):
        if False:
            print('Hello World!')
        import numpy as np
        import pandas as pd
        num_rows = 100000
        num_columns = 134
        data = np.zeros((num_rows, num_columns))
        columns = map(str, range(num_columns))
        df = SparkSession(self.sc).createDataFrame(pd.DataFrame(data, columns=columns))
        actual = CPickleSerializer().loads(df.rdd.map(list)._jrdd.first())
        expected = [list(data[0])]
        self.assertEqual(expected, actual)

    def test_sortByKey_uses_all_partitions_not_only_first_and_last(self):
        if False:
            return 10
        seq = [(i * 59 % 101, i) for i in range(101)]
        rdd = self.sc.parallelize(seq)
        for ascending in [True, False]:
            sort = rdd.sortByKey(ascending=ascending, numPartitions=5)
            self.assertEqual(sort.collect(), sorted(seq, reverse=not ascending))
            sizes = sort.glom().map(len).collect()
            for size in sizes:
                self.assertGreater(size, 0)

    def test_pipe_functions(self):
        if False:
            while True:
                i = 10
        data = ['1', '2', '3']
        rdd = self.sc.parallelize(data)
        with QuietTest(self.sc):
            self.assertEqual([], rdd.pipe('java').collect())
            self.assertRaises(Py4JJavaError, rdd.pipe('java', checkCode=True).collect)
        result = rdd.pipe('cat').collect()
        result.sort()
        for (x, y) in zip(data, result):
            self.assertEqual(x, y)
        self.assertRaises(Py4JJavaError, rdd.pipe('grep 4', checkCode=True).collect)
        self.assertEqual([], rdd.pipe('grep 4').collect())

    def test_pipe_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        data = ['测试', '1']
        rdd = self.sc.parallelize(data)
        result = rdd.pipe('cat').collect()
        self.assertEqual(data, result)

    def test_stopiteration_in_user_code(self):
        if False:
            while True:
                i = 10

        def stopit(*x):
            if False:
                for i in range(10):
                    print('nop')
            raise StopIteration()
        seq_rdd = self.sc.parallelize(range(10))
        keyed_rdd = self.sc.parallelize(((x % 2, x) for x in range(10)))
        msg = "Caught StopIteration thrown from user's code; failing the task"
        self.assertRaisesRegex(Py4JJavaError, msg, seq_rdd.map(stopit).collect)
        self.assertRaisesRegex(Py4JJavaError, msg, seq_rdd.filter(stopit).collect)
        self.assertRaisesRegex(Py4JJavaError, msg, seq_rdd.foreach, stopit)
        self.assertRaisesRegex(Py4JJavaError, msg, seq_rdd.reduce, stopit)
        self.assertRaisesRegex(Py4JJavaError, msg, seq_rdd.fold, 0, stopit)
        self.assertRaisesRegex(Py4JJavaError, msg, seq_rdd.foreach, stopit)
        self.assertRaisesRegex(Py4JJavaError, msg, seq_rdd.cartesian(seq_rdd).flatMap(stopit).collect)
        self.assertRaisesRegex((Py4JJavaError, RuntimeError), msg, keyed_rdd.reduceByKeyLocally, stopit)
        self.assertRaisesRegex((Py4JJavaError, RuntimeError), msg, seq_rdd.aggregate, 0, stopit, lambda *x: 1)
        self.assertRaisesRegex((Py4JJavaError, RuntimeError), msg, seq_rdd.aggregate, 0, lambda *x: 1, stopit)

    def test_overwritten_global_func(self):
        if False:
            while True:
                i = 10
        global global_func
        self.assertEqual(self.sc.parallelize([1]).map(lambda _: global_func()).first(), 'Hi')
        global_func = lambda : 'Yeah'
        self.assertEqual(self.sc.parallelize([1]).map(lambda _: global_func()).first(), 'Yeah')

    def test_to_local_iterator_failure(self):
        if False:
            for i in range(10):
                print('nop')

        def fail(_):
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('local iterator error')
        rdd = self.sc.range(10).map(fail)
        with self.assertRaisesRegex(Exception, 'local iterator error'):
            for _ in rdd.toLocalIterator():
                pass

    def test_to_local_iterator_collects_single_partition(self):
        if False:
            print('Hello World!')

        def fail_last(x):
            if False:
                for i in range(10):
                    print('nop')
            if x == 9:
                raise RuntimeError('This should not be hit')
            return x
        rdd = self.sc.range(12, numSlices=4).map(fail_last)
        it = rdd.toLocalIterator()
        for i in range(4):
            self.assertEqual(i, next(it))

    def test_resourceprofile(self):
        if False:
            for i in range(10):
                print('nop')
        rp_builder = ResourceProfileBuilder()
        ereqs = ExecutorResourceRequests().cores(2).memory('6g').memoryOverhead('1g')
        ereqs.pysparkMemory('2g').resource('gpu', 2, 'testGpus', 'nvidia.com')
        treqs = TaskResourceRequests().cpus(2).resource('gpu', 2)

        def assert_request_contents(exec_reqs, task_reqs):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(len(exec_reqs), 5)
            self.assertEqual(exec_reqs['cores'].amount, 2)
            self.assertEqual(exec_reqs['memory'].amount, 6144)
            self.assertEqual(exec_reqs['memoryOverhead'].amount, 1024)
            self.assertEqual(exec_reqs['pyspark.memory'].amount, 2048)
            self.assertEqual(exec_reqs['gpu'].amount, 2)
            self.assertEqual(exec_reqs['gpu'].discoveryScript, 'testGpus')
            self.assertEqual(exec_reqs['gpu'].resourceName, 'gpu')
            self.assertEqual(exec_reqs['gpu'].vendor, 'nvidia.com')
            self.assertEqual(len(task_reqs), 2)
            self.assertEqual(task_reqs['cpus'].amount, 2.0)
            self.assertEqual(task_reqs['gpu'].amount, 2.0)
        assert_request_contents(ereqs.requests, treqs.requests)
        rp = rp_builder.require(ereqs).require(treqs).build
        assert_request_contents(rp.executorResources, rp.taskResources)
        rdd = self.sc.parallelize(range(10)).withResources(rp)
        return_rp = rdd.getResourceProfile()
        assert_request_contents(return_rp.executorResources, return_rp.taskResources)
        rddWithoutRp = self.sc.parallelize(range(10))
        self.assertEqual(rddWithoutRp.getResourceProfile(), None)

    def test_multiple_group_jobs(self):
        if False:
            while True:
                i = 10
        import threading
        group_a = 'job_ids_to_cancel'
        group_b = 'job_ids_to_run'
        threads = []
        thread_ids = range(4)
        thread_ids_to_cancel = [i for i in thread_ids if i % 2 == 0]
        thread_ids_to_run = [i for i in thread_ids if i % 2 != 0]
        is_job_cancelled = [False for _ in thread_ids]

        def run_job(job_group, index):
            if False:
                i = 10
                return i + 15
            '\n            Executes a job with the group ``job_group``. Each job waits for 3 seconds\n            and then exits.\n            '
            try:
                self.sc.parallelize([15]).map(lambda x: time.sleep(x)).collectWithJobGroup(job_group, 'test rdd collect with setting job group')
                is_job_cancelled[index] = False
            except Exception:
                is_job_cancelled[index] = True
        run_job(group_a, 0)
        self.assertFalse(is_job_cancelled[0])
        for i in thread_ids_to_cancel:
            t = threading.Thread(target=run_job, args=(group_a, i))
            t.start()
            threads.append(t)
        for i in thread_ids_to_run:
            t = threading.Thread(target=run_job, args=(group_b, i))
            t.start()
            threads.append(t)
        time.sleep(3)
        self.sc.cancelJobGroup(group_a)
        for t in threads:
            t.join()
        for i in thread_ids_to_cancel:
            self.assertTrue(is_job_cancelled[i], 'Thread {i}: Job in group A was not cancelled.'.format(i=i))
        for i in thread_ids_to_run:
            self.assertFalse(is_job_cancelled[i], 'Thread {i}: Job in group B did not succeeded.'.format(i=i))
if __name__ == '__main__':
    import unittest
    from pyspark.tests.test_rdd import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
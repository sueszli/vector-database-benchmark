"""Tests for tf.data service coordinated reads."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class CoordinatedReadTest(data_service_test_base.TestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(num_workers=[1, 3], num_consumers=[1, 2, 5])))
    def testBasic(self, num_workers, num_consumers):
        if False:
            for i in range(10):
                print('nop')
        cluster = data_service_test_base.TestCluster(num_workers=num_workers)
        ds = self.make_coordinated_read_dataset(cluster, num_consumers)
        get_next = self.getNext(ds, requires_initialization=True)
        results = [self.evaluate(get_next()) for _ in range(100)]
        self.checkCoordinatedReadGroups(results, num_consumers)
        cluster.stop_workers()

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testConsumerRestart(self):
        if False:
            i = 10
            return i + 15
        cluster = data_service_test_base.TestCluster(num_workers=1)
        num_consumers = 3
        ds = self.make_coordinated_read_dataset(cluster, num_consumers)
        get_next = self.getNext(ds, requires_initialization=True)
        _ = [self.evaluate(get_next()) for _ in range(20)]
        ds2 = self.make_coordinated_read_dataset(cluster, num_consumers)
        with self.assertRaisesRegex(errors.FailedPreconditionError, 'current round has already reached'):
            get_next_ds2 = self.getNext(ds2, requires_initialization=True)
            _ = [self.evaluate(get_next_ds2()) for _ in range(20)]
        cluster.stop_workers()

    @combinations.generate(test_base.default_test_combinations())
    def testBucketizing(self):
        if False:
            while True:
                i = 10
        cluster = data_service_test_base.TestCluster(num_workers=4)
        num_elements = 100
        low_bucket_max = 30
        mid_bucket_max = 60
        bucket_boundaries = [low_bucket_max, mid_bucket_max]
        batch_size = 10
        num_consumer_hosts = 3
        replicas_per_consumer_host = 5
        num_consumers = num_consumer_hosts * replicas_per_consumer_host
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        ds = dataset_ops.Dataset.range(num_elements, output_type=dtypes.int32)
        ds = ds.shuffle(num_elements)
        ds = ds.repeat()
        ds = ds.apply(grouping.bucket_by_sequence_length(lambda x: x, bucket_boundaries, bucket_batch_sizes, drop_remainder=True))
        ds = ds.apply(grouping.group_by_window(lambda x: math_ops.cast(x[1], dtypes.int64), lambda _, x: dataset_ops.Dataset.from_tensors(x), window_size=num_consumers))
        ds = ds.flat_map(lambda x: x)
        host_datasets = []
        for host_index in range(num_consumer_hosts):
            per_replica_datasets = []
            for i in range(replicas_per_consumer_host):
                consumer_index = host_index * replicas_per_consumer_host + i
                per_replica_datasets.append(self.make_distributed_dataset(ds, cluster, job_name='test', consumer_index=consumer_index, num_consumers=num_consumers))
            host_dataset = dataset_ops.Dataset.from_tensor_slices(per_replica_datasets)
            host_dataset = host_dataset.interleave(lambda x: x, cycle_length=len(per_replica_datasets), num_parallel_calls=len(per_replica_datasets), deterministic=True)
            host_datasets.append(host_dataset)
        ds = dataset_ops.Dataset.from_tensor_slices(host_datasets)
        ds = ds.interleave(lambda x: x, block_length=replicas_per_consumer_host, cycle_length=len(host_datasets), num_parallel_calls=len(host_datasets), deterministic=True)
        num_rounds = 4
        get_next = self.getNext(ds, requires_initialization=True)
        results = []
        for i in range(num_rounds * num_consumers):
            results.append(self.evaluate(get_next()))

        def get_bucket(elem):
            if False:
                for i in range(10):
                    print('nop')
            bucket_ind = 0
            while bucket_ind < len(bucket_boundaries) and elem >= bucket_boundaries[bucket_ind]:
                bucket_ind += 1
            return bucket_ind
        for i in range(0, len(results), num_consumers):
            batches = results[num_consumers * i:num_consumers * (i + 1)]
            bucket_inds = [get_bucket(batch[0]) for batch in batches]
            for bucket_ind in bucket_inds[1:]:
                self.assertEqual(bucket_inds[0], bucket_ind, 'Batches: {}, Buckets: {}'.format(batches, bucket_inds))

    @combinations.generate(test_base.v1_only_combinations())
    def testFiniteV1(self):
        if False:
            for i in range(10):
                print('nop')
        cluster = data_service_test_base.TestCluster(num_workers=1)
        num_elements = 100
        ds = dataset_ops.Dataset.range(num_elements)
        ds = self.make_distributed_dataset(ds, cluster, job_name='test', consumer_index=0, num_consumers=1)
        with self.assertRaisesRegex(errors.FailedPreconditionError, 'Encountered end of sequence on a round-robin read iterator'):
            self.getDatasetOutput(ds)

    @combinations.generate(test_base.v2_only_combinations())
    def testFiniteV2(self):
        if False:
            while True:
                i = 10
        cluster = data_service_test_base.TestCluster(num_workers=1)
        num_elements = 100
        ds = dataset_ops.Dataset.range(num_elements)
        ds = self.make_distributed_dataset(ds, cluster, job_name='test', consumer_index=0, num_consumers=1)
        with self.assertRaisesRegex(errors.FailedPreconditionError, 'Round robin reads require that the input dataset has infinite cardinality, but the dataset has cardinality ' + str(num_elements)):
            self.getDatasetOutput(ds)

    @combinations.generate(test_base.v2_only_combinations())
    def testCardinality(self):
        if False:
            print('Hello World!')
        cluster = data_service_test_base.TestCluster(num_workers=1)
        ds = self.make_distributed_dataset(dataset_ops.Dataset.range(10).repeat(), cluster, job_name='test', consumer_index=0, num_consumers=2)
        self.assertEqual(self.evaluate(ds.cardinality()), dataset_ops.INFINITE)
if __name__ == '__main__':
    test.main()
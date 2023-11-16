import unittest

class TestClusterPartition(unittest.TestCase):

    def test_cluster_partition(self):
        if False:
            while True:
                i = 10
        clusters = [(5, 8), (1, 8), (4, 8), (16, 8), (2, 8), (3, 8)]
        from paddle.distributed.auto_parallel.static.tuner.rule_based_tuner import ClusterPartitionUtil
        device_meshes = []
        for cluster in clusters:
            n = cluster[0]
            m = cluster[1]
            device_mesh = ClusterPartitionUtil.partition_cluster(n, m)
            device_meshes.append(device_mesh)
if __name__ == '__main__':
    unittest.main()
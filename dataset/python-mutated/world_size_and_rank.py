import unittest
import paddle.distributed as dist

class TestWorldSizeAndRankAPI(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._num_of_ranks = 2
        self._subgroup_ranks = [0, 1]
        dist.init_parallel_env()
        self._subgroup = dist.new_group(self._subgroup_ranks)
        self._global_rank = dist.get_rank()

    def test_default_env_world_size(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(dist.get_world_size(), self._num_of_ranks)

    def test_given_group_world_size(self):
        if False:
            return 10
        world_size = 2 if self._global_rank in self._subgroup_ranks else -1
        self.assertEqual(dist.get_world_size(self._subgroup), world_size)

    def test_given_group_rank(self):
        if False:
            while True:
                i = 10
        rank = self._subgroup_ranks.index(self._global_rank) if self._global_rank in self._subgroup_ranks else -1
        self.assertEqual(dist.get_rank(self._subgroup), rank)
if __name__ == '__main__':
    unittest.main()
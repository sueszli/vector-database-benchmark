import collections
import unittest
import numpy as np
from fairseq.data import ListDataset, ResamplingDataset

class TestResamplingDataset(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.strings = ['ab', 'c', 'def', 'ghij']
        self.weights = [4.0, 2.0, 7.0, 1.5]
        self.size_ratio = 2
        self.dataset = ListDataset(self.strings, np.array([len(s) for s in self.strings]))

    def _test_common(self, resampling_dataset, iters):
        if False:
            i = 10
            return i + 15
        assert len(self.dataset) == len(self.strings) == len(self.weights)
        assert len(resampling_dataset) == self.size_ratio * len(self.strings)
        results = {'ordered_by_size': True, 'max_distribution_diff': 0.0}
        totalfreqs = 0
        freqs = collections.defaultdict(int)
        for epoch_num in range(iters):
            resampling_dataset.set_epoch(epoch_num)
            indices = resampling_dataset.ordered_indices()
            assert len(indices) == len(resampling_dataset)
            prev_size = -1
            for i in indices:
                cur_size = resampling_dataset.size(i)
                assert resampling_dataset[i] == resampling_dataset[i]
                assert cur_size == len(resampling_dataset[i])
                freqs[resampling_dataset[i]] += 1
                totalfreqs += 1
                if prev_size > cur_size:
                    results['ordered_by_size'] = False
                prev_size = cur_size
        assert set(freqs.keys()) == set(self.strings)
        for (s, weight) in zip(self.strings, self.weights):
            freq = freqs[s] / totalfreqs
            expected_freq = weight / sum(self.weights)
            results['max_distribution_diff'] = max(results['max_distribution_diff'], abs(expected_freq - freq))
        return results

    def test_resampling_dataset_batch_by_size_false(self):
        if False:
            print('Hello World!')
        resampling_dataset = ResamplingDataset(self.dataset, self.weights, size_ratio=self.size_ratio, batch_by_size=False, seed=0)
        results = self._test_common(resampling_dataset, iters=1000)
        assert not results['ordered_by_size']
        assert results['max_distribution_diff'] < 0.02

    def test_resampling_dataset_batch_by_size_true(self):
        if False:
            while True:
                i = 10
        resampling_dataset = ResamplingDataset(self.dataset, self.weights, size_ratio=self.size_ratio, batch_by_size=True, seed=0)
        results = self._test_common(resampling_dataset, iters=1000)
        assert results['ordered_by_size']
        assert results['max_distribution_diff'] < 0.02
if __name__ == '__main__':
    unittest.main()
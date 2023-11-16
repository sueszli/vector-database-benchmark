import unittest
import tests.utils as test_utils
import torch
from fairseq.data import TokenBlockDataset

class TestTokenBlockDataset(unittest.TestCase):

    def _build_dataset(self, data, **kwargs):
        if False:
            return 10
        sizes = [len(x) for x in data]
        underlying_ds = test_utils.TestDataset(data)
        return TokenBlockDataset(underlying_ds, sizes, **kwargs)

    def test_eos_break_mode(self):
        if False:
            i = 10
            return i + 15
        data = [torch.tensor([5, 4, 3, 2, 1], dtype=torch.long), torch.tensor([1], dtype=torch.long), torch.tensor([8, 7, 6, 1], dtype=torch.long)]
        ds = self._build_dataset(data, block_size=None, pad=0, eos=1, break_mode='eos')
        self.assertEqual(ds[0].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [1])
        self.assertEqual(ds[2].tolist(), [8, 7, 6, 1])
        data = [torch.tensor([5, 4, 3, 2, 1], dtype=torch.long), torch.tensor([8, 7, 6, 1], dtype=torch.long), torch.tensor([1], dtype=torch.long)]
        ds = self._build_dataset(data, block_size=None, pad=0, eos=1, break_mode='eos')
        self.assertEqual(ds[0].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [8, 7, 6, 1])
        self.assertEqual(ds[2].tolist(), [1])

    def test_block_break_mode(self):
        if False:
            i = 10
            return i + 15
        data = [torch.tensor([5, 4, 3, 2, 1], dtype=torch.long), torch.tensor([8, 7, 6, 1], dtype=torch.long), torch.tensor([9, 1], dtype=torch.long)]
        ds = self._build_dataset(data, block_size=3, pad=0, eos=1, break_mode='none')
        self.assertEqual(ds[0].tolist(), [5, 4, 3])
        self.assertEqual(ds[1].tolist(), [2, 1, 8])
        self.assertEqual(ds[2].tolist(), [7, 6, 1])
        self.assertEqual(ds[3].tolist(), [9, 1])

    def test_complete_break_mode(self):
        if False:
            while True:
                i = 10
        data = [torch.tensor([5, 4, 3, 2, 1], dtype=torch.long), torch.tensor([8, 7, 6, 1], dtype=torch.long), torch.tensor([9, 1], dtype=torch.long)]
        ds = self._build_dataset(data, block_size=6, pad=0, eos=1, break_mode='complete')
        self.assertEqual(ds[0].tolist(), [5, 4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [8, 7, 6, 1, 9, 1])
        data = [torch.tensor([4, 3, 2, 1], dtype=torch.long), torch.tensor([5, 1], dtype=torch.long), torch.tensor([1], dtype=torch.long), torch.tensor([6, 1], dtype=torch.long)]
        ds = self._build_dataset(data, block_size=3, pad=0, eos=1, break_mode='complete')
        self.assertEqual(ds[0].tolist(), [4, 3, 2, 1])
        self.assertEqual(ds[1].tolist(), [5, 1, 1])
        self.assertEqual(ds[2].tolist(), [6, 1])

    def test_4billion_tokens(self):
        if False:
            return 10
        'Regression test for numpy type promotion issue https://github.com/numpy/numpy/issues/5745'
        data = [torch.tensor(list(range(10000)), dtype=torch.long)] * 430000
        ds = self._build_dataset(data, block_size=6, pad=0, eos=1, break_mode='complete')
        ds[-1]
        (start, end) = ds.slice_indices[-1]
        assert end > 4294967295
        assert not isinstance(end + 1, float)
if __name__ == '__main__':
    unittest.main()
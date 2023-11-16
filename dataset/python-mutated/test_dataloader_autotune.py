import json
import os
import sys
import tempfile
import unittest
import warnings
import numpy as np
import paddle
from paddle import nn
from paddle.io import DataLoader, Dataset

class RandomDataset(Dataset):

    def __init__(self, num_samples):
        if False:
            print('Hello World!')
        self.num_samples = num_samples

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        image = np.random.random([10]).astype('float32')
        label = np.random.randint(0, 10 - 1, (1,)).astype('int64')
        return (image, label)

    def __len__(self):
        if False:
            return 10
        return self.num_samples

class SimpleNet(nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, image):
        if False:
            return 10
        return self.fc(image)

class TestAutoTune(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.batch_size = 1
        self.dataset = RandomDataset(10)

    def test_dataloader_use_autotune(self):
        if False:
            i = 10
            return i + 15
        paddle.incubate.autotune.set_config(config={'dataloader': {'enable': True, 'tuning_steps': 1}})
        loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader_disable_autotune(self):
        if False:
            i = 10
            return i + 15
        config = {'dataloader': {'enable': False, 'tuning_steps': 1}}
        tfile = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        json.dump(config, tfile)
        tfile.close()
        paddle.incubate.autotune.set_config(tfile.name)
        os.remove(tfile.name)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=2)
        if sys.platform == 'darwin' or sys.platform == 'win32':
            self.assertEqual(loader.num_workers, 0)
        else:
            self.assertEqual(loader.num_workers, 2)

    def test_distributer_batch_sampler_autotune(self):
        if False:
            print('Hello World!')
        paddle.incubate.autotune.set_config(config={'dataloader': {'enable': True, 'tuning_steps': 1}})
        batch_sampler = paddle.io.DistributedBatchSampler(self.dataset, batch_size=self.batch_size)
        loader = DataLoader(self.dataset, batch_sampler=batch_sampler, num_workers=2)

class TestAutoTuneAPI(unittest.TestCase):

    def test_set_config_warnings(self):
        if False:
            return 10
        with warnings.catch_warnings(record=True) as w:
            config = {'kernel': {'enable': 1, 'tuning_range': True}}
            tfile = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            json.dump(config, tfile)
            tfile.close()
            paddle.incubate.autotune.set_config(tfile.name)
            os.remove(tfile.name)
            self.assertTrue(len(w) == 2)
if __name__ == '__main__':
    unittest.main()
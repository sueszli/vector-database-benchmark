import os
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn as nn
from torch.distributed._tools import MemoryTracker
import unittest

class TestMemoryTracker(TestCase):

    @unittest.skipIf(not TEST_CUDA, 'no cuda')
    def test_local_model(self):
        if False:
            return 10
        '\n        Minimal test case to check the memory tracker can collect the expected\n        memory stats at operator level, as well as can print the summary result\n        without crash.\n        '
        torch.manual_seed(0)
        model = nn.Sequential(nn.Sequential(nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=False), nn.AdaptiveAvgPool2d(output_size=(1, 1))), nn.Flatten(start_dim=1), nn.Sequential(nn.Linear(64, 2), nn.ReLU(inplace=True))).cuda()
        tracker = MemoryTracker()
        tracker.start_monitor(model)
        x = torch.randn(size=(2, 3, 224, 224), device=torch.device('cuda'))
        target = torch.LongTensor([0, 1]).cuda()
        criterion = nn.CrossEntropyLoss()
        criterion(model(x), target).backward()
        self.assertTrue(len(tracker._hooks) > 0)
        tracker.stop()
        self.assertTrue(len(tracker._hooks) == 0)
        path = 'memory.trace'
        tracker.save_stats(path)
        tracker.load(path)
        tracker.summary()
        if os.path.exists(path):
            os.remove(path)
        self.assertTrue(tracker._op_index > 0)
        self.assertTrue(len(tracker._operator_names) > 0)
        self.assertEqual(len(tracker.memories_allocated), tracker._op_index)
        self.assertEqual(len(tracker.memories_active), tracker._op_index)
        self.assertEqual(len(tracker.memories_reserved), tracker._op_index)
        self.assertTrue(len(tracker._markers) == 2)
        self.assertTrue(tracker._cur_module_name != '')
        self.assertTrue(hasattr(tracker, '_num_cuda_retries'))
if __name__ == '__main__':
    run_tests()
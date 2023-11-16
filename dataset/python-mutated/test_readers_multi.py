from .base import TestCase
import os
import shutil
import time
from django.conf import settings
import whisper
import gzip
from graphite.readers import WhisperReader, MultiReader
from graphite.node import LeafNode

class MultiReaderTests(TestCase):
    start_ts = 0
    hostcpu = os.path.join(settings.WHISPER_DIR, 'hosts/hostname/cpu.wsp')
    worker1 = hostcpu.replace('hostname', 'worker1')
    worker2 = hostcpu.replace('hostname', 'worker2')
    worker3 = hostcpu.replace('hostname', 'worker3')
    worker4 = hostcpu.replace('hostname', 'worker4')
    worker4 = worker4.replace('cpu.wsp', 'cpu.wsp.gz')

    def create_whisper_hosts(self):
        if False:
            i = 10
            return i + 15
        self.start_ts = int(time.time())
        try:
            os.makedirs(self.worker1.replace('cpu.wsp', ''))
            os.makedirs(self.worker2.replace('cpu.wsp', ''))
            os.makedirs(self.worker3.replace('cpu.wsp', ''))
            os.makedirs(self.worker4.replace('cpu.wsp.gz', ''))
        except OSError:
            pass
        whisper.create(self.worker1, [(1, 60)])
        whisper.create(self.worker2, [(1, 60)])
        open(self.worker3, 'a').close()
        whisper.update(self.worker1, 1, self.start_ts)
        whisper.update(self.worker2, 2, self.start_ts)
        with open(self.worker1, 'rb') as f_in, gzip.open(self.worker4, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    def wipe_whisper_hosts(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            os.remove(self.worker1)
            os.remove(self.worker2)
            os.remove(self.worker3)
            os.remove(self.worker4)
            shutil.rmtree(os.path.join(settings.WHISPER_DIR, 'hosts'))
        except OSError:
            pass

    def test_MultiReader_init(self):
        if False:
            return 10
        self.create_whisper_hosts()
        self.addCleanup(self.wipe_whisper_hosts)
        wr1 = WhisperReader(self.worker1, 'hosts.worker1.cpu')
        node1 = LeafNode('hosts.worker1.cpu', wr1)
        wr2 = WhisperReader(self.worker2, 'hosts.worker2.cpu')
        node2 = LeafNode('hosts.worker2.cpu', wr2)
        reader = MultiReader([node1, node2])
        self.assertIsNotNone(reader)

    def test_MultiReader_get_intervals(self):
        if False:
            return 10
        self.create_whisper_hosts()
        self.addCleanup(self.wipe_whisper_hosts)
        wr1 = WhisperReader(self.worker1, 'hosts.worker1.cpu')
        node1 = LeafNode('hosts.worker1.cpu', wr1)
        wr2 = WhisperReader(self.worker2, 'hosts.worker2.cpu')
        node2 = LeafNode('hosts.worker2.cpu', wr2)
        reader = MultiReader([node1, node2])
        intervals = reader.get_intervals()
        for interval in intervals:
            self.assertIn(int(interval.start), [self.start_ts - 60, self.start_ts - 60 - 1])
            self.assertIn(int(interval.end), [self.start_ts, self.start_ts - 1])
            self.assertIn(int(interval.end - interval.start), [59, 60])

    def test_MultiReader_fetch(self):
        if False:
            i = 10
            return i + 15
        self.create_whisper_hosts()
        self.addCleanup(self.wipe_whisper_hosts)
        wr1 = WhisperReader(self.worker1, 'hosts.worker1.cpu')
        node1 = LeafNode('hosts.worker1.cpu', wr1)
        wr2 = WhisperReader(self.worker2, 'hosts.worker2.cpu')
        node2 = LeafNode('hosts.worker2.cpu', wr2)
        reader = MultiReader([node1, node2])
        results = reader.fetch(self.start_ts - 5, self.start_ts)
        (_, values) = results
        self.assertEqual(values, [None, None, None, None, 1.0])

    def test_MultiReader_merge_normal(self):
        if False:
            print('Hello World!')
        results1 = ((1496252939, 1496252944, 1), [None, None, None, None, 1.0])
        results2 = ((1496252939, 1496252944, 1), [1.0, 1.0, 1.0, 1.0, 1.0])
        wr1 = WhisperReader(self.worker1, 'hosts.worker1.cpu')
        node1 = LeafNode('hosts.worker1.cpu', wr1)
        reader = MultiReader([node1])
        (_, values) = reader.merge(results1, results2)
        self.assertEqual(values, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_MultiReader_merge_results1_finer_than_results2(self):
        if False:
            print('Hello World!')
        results1 = ((1496252939, 1496252944, 1), [1.0, None, None, None, 1.0])
        results2 = ((1496252939, 1496252949, 5), [1.0, 1.0])
        wr1 = WhisperReader(self.worker1, 'hosts.worker1.cpu')
        node1 = LeafNode('hosts.worker1.cpu', wr1)
        reader = MultiReader([node1])
        (_, values) = reader.merge(results1, results2)
        self.assertEqual(values, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_MultiReader_merge_results2_finer_than_results1(self):
        if False:
            print('Hello World!')
        results1 = ((1496252939, 1496252949, 5), [1.0, 1.0])
        results2 = ((1496252939, 1496252944, 1), [1.0, None, 1.0, None, 1.0])
        wr1 = WhisperReader(self.worker1, 'hosts.worker1.cpu')
        node1 = LeafNode('hosts.worker1.cpu', wr1)
        reader = MultiReader([node1])
        (_, values) = reader.merge(results1, results2)
        self.assertEqual(values, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_MultiReader_merge_results1_missing_some(self):
        if False:
            return 10
        results1 = ((1496252939, 1496252944, 1), [1.0, None, None, None, 1.0])
        results2 = ((1496252939, 1496252949, 1), [1.0, 1.0])
        wr1 = WhisperReader(self.worker1, 'hosts.worker1.cpu')
        node1 = LeafNode('hosts.worker1.cpu', wr1)
        reader = MultiReader([node1])
        (_, values) = reader.merge(results1, results2)
        self.assertEqual(values, [1.0, 1.0, None, None, 1.0, None, None, None, None, None])
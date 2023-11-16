from .base import TestCase
import os
from os.path import join, isdir
import rrdtool
import shutil
import six
import time
from django.conf import settings
from graphite.readers import RRDReader
from graphite.wsgi import application

class RRDReaderTests(TestCase):
    test_dir = join(settings.RRD_DIR)
    start_ts = 0
    step = 60
    points = 100
    hostcpu = os.path.join(test_dir, 'hosts/worker1/cpu.rrd')

    def create_rrd(self):
        if False:
            print('Hello World!')
        if not isdir(self.test_dir):
            os.makedirs(self.test_dir)
        try:
            os.makedirs(self.hostcpu.replace('cpu.rrd', ''))
        except OSError:
            pass
        self.start_ts = int(time.time())
        rrdtool.create(self.hostcpu, '--start', str(self.start_ts), '--step', str(self.step), 'RRA:AVERAGE:0.5:1:{}'.format(self.points), 'DS:cpu:GAUGE:60:U:U')

    def wipe_rrd(self):
        if False:
            print('Hello World!')
        try:
            shutil.rmtree(self.test_dir)
        except OSError:
            pass

    def test_RRDReader_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.create_rrd()
        self.addCleanup(self.wipe_rrd)
        reader = RRDReader(self.hostcpu, 'cpu')
        self.assertIsNotNone(reader)

    def test_RRDReader_convert_fs_path(self):
        if False:
            i = 10
            return i + 15
        path = RRDReader._convert_fs_path(six.u(self.hostcpu))
        self.assertIsInstance(path, str)

    def test_RRDReader_get_intervals(self):
        if False:
            while True:
                i = 10
        self.create_rrd()
        self.addCleanup(self.wipe_rrd)
        reader = RRDReader(self.hostcpu, 'cpu')
        for interval in reader.get_intervals():
            self.assertAlmostEqual(interval.start, self.start_ts - self.points * self.step, delta=2)
            self.assertAlmostEqual(interval.end, self.start_ts, delta=2)

    def test_RRDReader_fetch(self):
        if False:
            for i in range(10):
                print('nop')
        self.create_rrd()
        self.addCleanup(self.wipe_rrd)
        for ts in range(self.start_ts + 60, self.start_ts + 10 * self.step, self.step):
            rrdtool.update(self.hostcpu, '{}:42'.format(ts))
        reader = RRDReader(self.hostcpu, 'cpu')
        (time_info, values) = reader.fetch(self.start_ts + self.step, self.start_ts + self.step * 2)
        self.assertEqual(list(values), [42.0])

    def test_RRDReader_get_datasources(self):
        if False:
            print('Hello World!')
        self.create_rrd()
        self.addCleanup(self.wipe_rrd)
        datasource = RRDReader.get_datasources(self.hostcpu)
        self.assertEqual(datasource, ['cpu'])

    def test_RRDReader_get_retention(self):
        if False:
            return 10
        self.create_rrd()
        self.addCleanup(self.wipe_rrd)
        retentions = RRDReader.get_retention(self.hostcpu)
        self.assertEqual(retentions, self.points * self.step)
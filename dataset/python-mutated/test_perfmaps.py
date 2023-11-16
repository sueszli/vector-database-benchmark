import os
import sys
import unittest
from _testinternalcapi import perf_map_state_teardown, write_perf_map_entry
if sys.platform != 'linux':
    raise unittest.SkipTest('Linux only')

class TestPerfMapWriting(unittest.TestCase):

    def test_write_perf_map_entry(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(write_perf_map_entry(4660, 5678, 'entry1'), 0)
        self.assertEqual(write_perf_map_entry(9029, 6789, 'entry2'), 0)
        with open(f'/tmp/perf-{os.getpid()}.map') as f:
            perf_file_contents = f.read()
            self.assertIn('1234 162e entry1', perf_file_contents)
            self.assertIn('2345 1a85 entry2', perf_file_contents)
        perf_map_state_teardown()
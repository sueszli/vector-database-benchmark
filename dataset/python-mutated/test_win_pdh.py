import pytest
import salt.utils.win_pdh as win_pdh
from tests.support.mock import patch
from tests.support.unit import TestCase
try:
    import pywintypes
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

@pytest.mark.skipif(not HAS_WIN32, reason='Requires pywin32')
@pytest.mark.skip_unless_on_windows
class WinPdhTestCase(TestCase):

    @pytest.mark.slow_test
    def test_list_objects(self):
        if False:
            i = 10
            return i + 15
        known_objects = ['Cache', 'Memory', 'Process', 'Processor', 'System']
        objects = win_pdh.list_objects()
        for item in known_objects:
            self.assertTrue(item in objects)

    def test_list_counters(self):
        if False:
            print('Hello World!')
        counters = win_pdh.list_counters('Processor')
        known_counters = ['% Processor Time', '% User Time', '% DPC Time']
        for item in known_counters:
            self.assertTrue(item in counters)

    def test_list_instances(self):
        if False:
            print('Hello World!')
        instances = win_pdh.list_instances('Processor')
        known_instances = ['0', '_Total']
        for item in known_instances:
            self.assertTrue(item in instances)

    def test_build_counter_list(self):
        if False:
            print('Hello World!')
        counter_list = [('Memory', None, 'Available Bytes'), ('Paging File', '*', '% Usage'), ('Processor', '*', '% Processor Time'), ('Server', None, 'Work Item Shortages'), ('Server Work Queues', '*', 'Queue Length'), ('System', None, 'Context Switches/sec')]
        resulting_list = win_pdh.build_counter_list(counter_list)
        for counter in resulting_list:
            self.assertTrue(isinstance(counter, win_pdh.Counter))
        resulting_paths = []
        for counter in resulting_list:
            resulting_paths.append(counter.path)
        expected_paths = ['\\Memory\\Available Bytes', '\\Paging File(*)\\% Usage', '\\Processor(*)\\% Processor Time', '\\Server\\Work Item Shortages', '\\Server Work Queues(*)\\Queue Length', '\\System\\Context Switches/sec']
        self.assertEqual(resulting_paths, expected_paths)

    @pytest.mark.slow_test
    def test_get_all_counters(self):
        if False:
            for i in range(10):
                print('nop')
        results = win_pdh.get_all_counters('Processor')
        known_counters = ['\\Processor(*)\\% Processor Time', '\\Processor(*)\\% Idle Time', '\\Processor(*)\\DPC Rate', '\\Processor(*)\\% Privileged Time', '\\Processor(*)\\DPCs Queued/sec', '\\Processor(*)\\% Interrupt Time', '\\Processor(*)\\Interrupts/sec']
        for item in known_counters:
            self.assertTrue(item in results)

    @pytest.mark.slow_test
    def test_get_counters(self):
        if False:
            i = 10
            return i + 15
        counter_list = [('Memory', None, 'Available Bytes'), ('Paging File', '*', '% Usage'), ('Processor', '*', '% Processor Time'), ('Server', None, 'Work Item Shortages'), ('Server Work Queues', '*', 'Queue Length'), ('System', None, 'Context Switches/sec')]
        results = win_pdh.get_counters(counter_list)
        expected_counters = ['\\Memory\\Available Bytes', '\\Paging File(*)\\% Usage', '\\Processor(*)\\% Processor Time', '\\Server\\Work Item Shortages', '\\Server Work Queues(*)\\Queue Length', '\\System\\Context Switches/sec']
        for item in expected_counters:
            self.assertTrue(item in results)

    def test_get_counter(self):
        if False:
            while True:
                i = 10
        results = win_pdh.get_counter('Processor', '*', '% Processor Time')
        self.assertTrue('\\Processor(*)\\% Processor Time' in results)

    @patch('win32pdh.CollectQueryData')
    def test_get_counters_no_data_to_return(self, mock_query):
        if False:
            i = 10
            return i + 15
        mock_query.side_effect = pywintypes.error(-2147481643, 'CollectQueryData', 'No data to return.')
        counter_list = [('Memory', None, 'Available Bytes'), ('Paging File', '*', '% Usage'), ('Processor', '*', '% Processor Time'), ('Server', None, 'Work Item Shortages'), ('Server Work Queues', '*', 'Queue Length'), ('System', None, 'Context Switches/sec')]
        results = win_pdh.get_counters(counter_list)
        assert results == {}
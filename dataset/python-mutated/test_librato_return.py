"""
Tests for the librato returner
"""
import logging
import pytest
from salt.returners import librato_return
from tests.support.case import ShellCase
log = logging.getLogger(__name__)
MOCK_RET_OBJ = {'fun_args': [], 'return': {'test-return-state': {'comment': 'insertcommenthere', 'name': 'test-state-1', 'start_time': '01: 19: 51.105566', 'result': True, 'duration': 3.645, '__run_num__': 193, 'changes': {}, '__id__': 'test-return-state'}, 'test-return-state2': {'comment': 'insertcommenthere', 'name': 'test-state-2', 'start_time': '01: 19: 51.105566', 'result': False, 'duration': 3.645, '__run_num__': 194, 'changes': {}, '__id__': 'test-return-state'}}, 'retcode': 2, 'success': True, 'fun': 'state.highstate', 'id': 'Librato-Test', 'out': 'highstate'}

@pytest.mark.windows_whitelisted
class libratoTest(ShellCase):
    """
    Test the librato returner
    """

    def test_count_runtimes(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the calculations\n        '
        results = librato_return._calculate_runtimes(MOCK_RET_OBJ['return'])
        self.assertEqual(results['num_failed_states'], 1)
        self.assertEqual(results['num_passed_states'], 1)
        self.assertEqual(results['runtime'], 7.29)
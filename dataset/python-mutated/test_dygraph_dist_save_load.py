import os
import subprocess
import sys
import unittest

def strategy_test(saving, loading, gather_to):
    if False:
        return 10
    cmd = f'{sys.executable} dygraph_dist_save_load.py --test_case {saving}:{loading} --gather_to {gather_to}'
    p = subprocess.Popen(cmd.split())
    p.communicate()
    assert p.poll() == 0

class TestDistSaveLoad(unittest.TestCase):

    def test_dygraph_save_load_dp_sharding_stage2(self):
        if False:
            print('Hello World!')
        strategy_test('dp', 'sharding_stage2', 0)
        strategy_test('dp', 'sharding_stage2', 1)
        strategy_test('sharding_stage2', 'dp', 1)
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    unittest.main()
import os
import subprocess
import sys
import unittest

def strategy_test(saving, seed=1024, loading='static'):
    if False:
        for i in range(10):
            print('nop')
    cmd = f'{sys.executable} dygraph_save_for_auto_infer.py --test_case {saving}:{loading} --cmd main --seed {seed}'
    p = subprocess.Popen(cmd.split())
    p.communicate()
    assert p.poll() == 0

class TestHybrid(unittest.TestCase):

    def test_dygraph_save_load_dp_sharding_stage2(self):
        if False:
            while True:
                i = 10
        strategy_test('dp')
        strategy_test('mp')
        strategy_test('pp')

class TestSharding(unittest.TestCase):

    def test_dygraph_save_load_dp_sharding_stage2(self):
        if False:
            i = 10
            return i + 15
        strategy_test('sharding_stage2')
        strategy_test('sharding_stage3')

class TestSingleCard(unittest.TestCase):

    def test_dygraph_save_load_dp_sharding_stage2(self):
        if False:
            while True:
                i = 10
        strategy_test('single')
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    unittest.main()
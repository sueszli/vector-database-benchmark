import random
from subprocess import CalledProcessError
from unittest.mock import patch
from main import main

def myoutput(cmdline):
    if False:
        i = 10
        return i + 15
    print(cmdline)
    if random.random() < 0.9:
        return (str(random.random()), str(random.random()))
    else:
        raise CalledProcessError('a', 'b')

def test_integration():
    if False:
        for i in range(10):
            print('nop')
    with patch('evaluation.runzen', myoutput) as cmd:
        with patch('evaluation.subprocess.check_output', myoutput):
            main('gdax.BTC-ETH', 120)
import random
import runpy
import sys
import mockturtle
sys.modules['turtle'] = sys.modules['mockturtle']

def test_avoid_1():
    if False:
        for i in range(10):
            print('nop')
    random.seed(0)
    mockturtle.events.clear()
    mockturtle.events += [('timer',)] * 10
    mockturtle.events += [('key Up',)]
    mockturtle.events += [('timer',)] * 10
    mockturtle.events += [('key Down',)]
    mockturtle.events += [('timer', True)] * 1000
    runpy.run_module('freegames.avoid')

def test_avoid_2():
    if False:
        print('Hello World!')
    random.seed(0)
    mockturtle.events.clear()
    mockturtle.events += [('timer', True), ('key Up',), ('timer', True), ('key Down',)] * 1000
    runpy.run_module('freegames.avoid')
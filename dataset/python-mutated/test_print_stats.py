import numpy as np
import pytest
from neon.benchmark import Benchmark
print_stats = Benchmark.print_stats

class NotImplementedOutput(object):
    pass

class TestOutput(object):

    def __init(self):
        if False:
            print('Hello World!')
        self.display_output = ''

    def display(self, data):
        if False:
            i = 10
            return i + 15
        self.display_output = data
test_stats = {'feature_name': [10, 10.5, 11, 11.6]}

def test_empty_functions():
    if False:
        print('Hello World!')
    test_console = TestOutput()
    print_stats(test_stats, functions=[], output=test_console)
    assert all((s in test_console.display_output for s in ['Mean', 'Median', 'Amin', 'Amax', 'feature_name']))

def test_custom_functions():
    if False:
        i = 10
        return i + 15
    test_console = TestOutput()
    print_stats(test_stats, functions=[np.average], output=test_console)
    assert not any((s in test_console.display_output for s in ['Mean', 'Median', 'Amin', 'Amax']))
    assert 'Average' in test_console.display_output
    assert 'feature_name' in test_console.display_output

def test_output_not_implemented():
    if False:
        while True:
            i = 10
    test_console = NotImplementedOutput()
    with pytest.raises(TypeError):
        print_stats(test_stats, output=test_console)

def test_output_no_stats():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        print_stats(None)

def test_output_empty_stats():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        print_stats([])
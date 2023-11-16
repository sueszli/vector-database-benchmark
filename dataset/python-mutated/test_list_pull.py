from collections import namedtuple
from dagster._utils import list_pull

def test_list_pull():
    if False:
        for i in range(10):
            print('nop')
    assert list_pull([], 'foo') == []

def test_list_pull_dicts():
    if False:
        i = 10
        return i + 15
    test_data = [{'foo': 'bar1'}, {'foo': 'bar2'}]
    assert list_pull(test_data, 'foo') == ['bar1', 'bar2']

def test_pull_objects():
    if False:
        while True:
            i = 10
    TestObject = namedtuple('TestObject', 'bar')
    test_objs = [TestObject(bar=2), TestObject(bar=3)]
    assert list_pull(test_objs, 'bar') == [2, 3]
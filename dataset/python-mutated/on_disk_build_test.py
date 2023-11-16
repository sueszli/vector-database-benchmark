import os
import pytest
from annoy import AnnoyIndex

@pytest.fixture(scope='module', autouse=True)
def setUp():
    if False:
        print('Hello World!')
    if os.path.exists('on_disk.ann'):
        os.remove('on_disk.ann')

def add_items(i):
    if False:
        i = 10
        return i + 15
    i.add_item(0, [2, 2])
    i.add_item(1, [3, 2])
    i.add_item(2, [3, 3])

def check_nns(i):
    if False:
        print('Hello World!')
    assert i.get_nns_by_vector([4, 4], 3) == [2, 1, 0]
    assert i.get_nns_by_vector([1, 1], 3) == [0, 1, 2]
    assert i.get_nns_by_vector([4, 2], 3) == [1, 2, 0]

def test_on_disk():
    if False:
        i = 10
        return i + 15
    f = 2
    i = AnnoyIndex(f, 'euclidean')
    i.on_disk_build('on_disk.ann')
    add_items(i)
    i.build(10)
    check_nns(i)
    i.unload()
    i.load('on_disk.ann')
    check_nns(i)
    j = AnnoyIndex(f, 'euclidean')
    j.load('on_disk.ann')
    check_nns(j)
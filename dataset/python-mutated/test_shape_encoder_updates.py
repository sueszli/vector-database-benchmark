import numpy as np
from deeplake.core.meta.encode.shape import ShapeEncoder
from .common import assert_encoded

def test_update_no_change():
    if False:
        for i in range(10):
            print('nop')
    enc = ShapeEncoder([[101, 100, 1], [100, 101, 5]])
    enc[0] = (101, 100)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])
    enc[1] = (101, 100)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])
    enc[2] = (100, 101)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])
    enc[3] = (100, 101)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])
    enc[4] = (100, 101)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])
    enc[5] = (100, 101)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])
    assert enc.num_samples == 6

def test_update_squeeze_trivial():
    if False:
        while True:
            i = 10
    enc = ShapeEncoder([[28, 0, 2], [100, 100, 3], [28, 0, 5]])
    enc[3] = (28, 0)
    assert_encoded(enc, [[28, 0, 5]])
    assert enc.num_samples == 6

def test_update_squeeze_complex():
    if False:
        return 10
    enc = ShapeEncoder([[10, 10, 1], [28, 0, 2], [100, 100, 3], [28, 0, 5], [10, 10, 7]])
    enc[3] = (28, 0)
    assert_encoded(enc, [[10, 10, 1], [28, 0, 5], [10, 10, 7]])
    assert enc.num_samples == 8

def test_update_squeeze_up():
    if False:
        print('Hello World!')
    enc = ShapeEncoder([[28, 25, 0], [28, 28, 8], [0, 0, 9]])
    enc[9] = (28, 28)
    assert_encoded(enc, [[28, 25, 0], [28, 28, 9]])
    assert enc.num_samples == 10

def test_update_squeeze_down():
    if False:
        while True:
            i = 10
    enc = ShapeEncoder([[28, 25, 0], [28, 28, 8], [0, 0, 9]])
    enc[0] = (28, 28)
    assert_encoded(enc, [[28, 28, 8], [0, 0, 9]])
    assert enc.num_samples == 10

def test_update_move_up():
    if False:
        i = 10
        return i + 15
    enc = ShapeEncoder([[101, 100, 0], [100, 101, 5]])
    enc[1] = (101, 100)
    assert_encoded(enc, [[101, 100, 1], [100, 101, 5]])
    enc[2] = (101, 100)
    assert_encoded(enc, [[101, 100, 2], [100, 101, 5]])
    assert enc.num_samples == 6

def test_update_move_down():
    if False:
        return 10
    enc = ShapeEncoder([[101, 100, 5], [100, 101, 10]])
    enc[5] = (100, 101)
    assert_encoded(enc, [[101, 100, 4], [100, 101, 10]])
    enc[4] = (100, 101)
    assert_encoded(enc, [[101, 100, 3], [100, 101, 10]])
    enc[3] = (100, 101)
    assert_encoded(enc, [[101, 100, 2], [100, 101, 10]])
    assert enc.num_samples == 11

def test_update_replace():
    if False:
        print('Hello World!')
    enc = ShapeEncoder([[100, 100, 0]])
    enc[0] = (100, 101)
    assert enc.num_samples == 1

def test_update_split_up():
    if False:
        i = 10
        return i + 15
    enc = ShapeEncoder([[100, 101, 5]])
    enc[0] = (101, 100)
    assert_encoded(enc, [[101, 100, 0], [100, 101, 5]])

def test_update_split_down():
    if False:
        i = 10
        return i + 15
    enc = ShapeEncoder([[100, 101, 5]])
    enc[5] = (101, 100)
    assert_encoded(enc, [[100, 101, 4], [101, 100, 5]])

def test_update_split_middle():
    if False:
        for i in range(10):
            print('nop')
    enc = ShapeEncoder([[28, 0, 5]])
    enc[3] = (100, 100)
    assert_encoded(enc, [[28, 0, 2], [100, 100, 3], [28, 0, 5]])
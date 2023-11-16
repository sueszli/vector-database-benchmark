import multiprocessing
import pytest
from dramatiq.canteen import Canteen, canteen_add, canteen_get, canteen_try_init

def test_canteen_add_adds_paths():
    if False:
        i = 10
        return i + 15
    c = multiprocessing.Value(Canteen)
    with canteen_try_init(c):
        canteen_add(c, 'hello')
        canteen_add(c, 'there')
    assert canteen_get(c) == ['hello', 'there']

def test_canteen_add_fails_when_adding_too_many_paths():
    if False:
        print('Hello World!')
    c = Canteen()
    with pytest.raises(RuntimeError):
        for _ in range(1024):
            canteen_add(c, '0' * 1024)

def test_canteen_try_init_runs_at_most_once():
    if False:
        while True:
            i = 10
    c = multiprocessing.Value(Canteen)
    with canteen_try_init(c) as acquired:
        if acquired:
            canteen_add(c, 'hello')
    with canteen_try_init(c) as acquired:
        if acquired:
            canteen_add(c, 'goodbye')
    assert canteen_get(c) == ['hello']
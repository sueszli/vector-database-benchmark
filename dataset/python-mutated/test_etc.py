from pytest import mark
import zmq
only_bundled = mark.skipif(not hasattr(zmq, '_libzmq'), reason='bundled libzmq')

@mark.skipif('zmq.zmq_version_info() < (4, 1)')
def test_has():
    if False:
        while True:
            i = 10
    assert not zmq.has('something weird')

@only_bundled
def test_has_curve():
    if False:
        for i in range(10):
            print('nop')
    'bundled libzmq has curve support'
    assert zmq.has('curve')

@only_bundled
def test_has_ipc():
    if False:
        while True:
            i = 10
    'bundled libzmq has ipc support'
    assert zmq.has('ipc')
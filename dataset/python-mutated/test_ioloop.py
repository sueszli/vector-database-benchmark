import pytest
try:
    import tornado.ioloop
except ImportError:
    _tornado = False
else:
    _tornado = True

def setup():
    if False:
        return 10
    if not _tornado:
        pytest.skip('requires tornado')

def test_ioloop():
    if False:
        for i in range(10):
            print('nop')
    from zmq.eventloop import ioloop
    assert ioloop.IOLoop is tornado.ioloop.IOLoop
    assert ioloop.ZMQIOLoop is ioloop.IOLoop

def test_ioloop_install():
    if False:
        return 10
    from zmq.eventloop import ioloop
    with pytest.warns(DeprecationWarning):
        ioloop.install()
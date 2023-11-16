"""Tests for wiring causes no issues with queue.Queue from std lib."""
from pytest import fixture
from samples.wiring import queuemodule
from samples.wiring.container import Container

@fixture
def container():
    if False:
        return 10
    container = Container()
    yield container
    container.unwire()

def test_wire_queue(container: Container):
    if False:
        for i in range(10):
            print('nop')
    try:
        container.wire(modules=[queuemodule])
    except:
        raise
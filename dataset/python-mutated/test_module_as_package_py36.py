"""Tests for wiring to module as package."""
from pytest import fixture
from samples.wiring import module
from samples.wiring.service import Service
from samples.wiring.container import Container

@fixture
def container():
    if False:
        return 10
    container = Container()
    yield container
    container.unwire()

def test_module_as_package_wiring(container: Container):
    if False:
        i = 10
        return i + 15
    container.wire(packages=[module])
    assert isinstance(module.service, Service)
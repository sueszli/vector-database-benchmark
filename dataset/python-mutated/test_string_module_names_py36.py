"""Tests for string module and package names."""
from pytest import fixture
from samples.wiring import module
from samples.wiring.service import Service
from samples.wiring.container import Container
from samples.wiring.wire_relative_string_names import wire_with_relative_string_names

@fixture
def container():
    if False:
        return 10
    container = Container()
    yield container
    container.unwire()

def test_absolute_names(container: Container):
    if False:
        for i in range(10):
            print('nop')
    container.wire(modules=['samples.wiring.module'], packages=['samples.wiring.package'])
    service = module.test_function()
    assert isinstance(service, Service)
    from samples.wiring.package.subpackage.submodule import test_function
    service = test_function()
    assert isinstance(service, Service)

def test_relative_names_with_explicit_package(container: Container):
    if False:
        while True:
            i = 10
    container.wire(modules=['.module'], packages=['.package'], from_package='samples.wiring')
    service = module.test_function()
    assert isinstance(service, Service)
    from samples.wiring.package.subpackage.submodule import test_function
    service = test_function()
    assert isinstance(service, Service)

def test_relative_names_with_auto_package(container: Container):
    if False:
        return 10
    wire_with_relative_string_names(container)
    service = module.test_function()
    assert isinstance(service, Service)
    from samples.wiring.package.subpackage.submodule import test_function
    service = test_function()
    assert isinstance(service, Service)
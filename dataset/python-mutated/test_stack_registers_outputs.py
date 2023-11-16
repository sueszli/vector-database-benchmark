"""Verifies that a Stack always calls RegisterResourceOutputs even if
there are no outputs. This makes sure removing stack outputs from a
program actually deletes them from the stack.

Regresses https://github.com/pulumi/pulumi/issues/8273

"""
import pytest
from pulumi.runtime import settings, mocks
import pulumi
from copy import deepcopy

class MyMocks(pulumi.runtime.Mocks):

    def new_resource(self, args: pulumi.runtime.MockResourceArgs):
        if False:
            while True:
                i = 10
        raise Exception('new_resource')

    def call(self, args: pulumi.runtime.MockCallArgs):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('call')

class MyMonitor(mocks.MockMonitor):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.outputs = None

    def RegisterResourceOutputs(self, outputs):
        if False:
            print('Hello World!')
        self.outputs = outputs

@pytest.fixture
def my_mocks():
    if False:
        i = 10
        return i + 15
    settings.reset_options()
    old_settings = deepcopy(settings.SETTINGS)
    monitor = MyMonitor()
    mm = MyMocks()
    mocks.set_mocks(mm, preview=False, monitor=monitor)
    try:
        yield mm
    finally:
        settings.configure(old_settings)
        assert monitor.outputs is not None
        assert type(monitor.outputs.urn) == str

@pulumi.runtime.test
def test_stack_registers_outputs(my_mocks):
    if False:
        return 10
    pass
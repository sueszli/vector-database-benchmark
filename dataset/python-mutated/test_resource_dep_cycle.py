from typing import Optional
import pytest
from pulumi.runtime import settings, mocks
import pulumi

@pytest.mark.timeout(10)
@pulumi.runtime.test
def test_pulumi_does_not_hang_on_dependency_cycle(my_mocks):
    if False:
        for i in range(10):
            print('nop')
    c = MockComponentResource(name='c')
    r = MockResource(name='r', input1=c.output1, opts=pulumi.ResourceOptions(parent=c))
    return pulumi.Output.all(c.urn, r.urn).apply(print)

class MockResource(pulumi.CustomResource):

    def __init__(self, name: str, input1: pulumi.Input[str], opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            i = 10
            return i + 15
        super().__init__('python:test_resource_dep_cycle:MockResource', name, {'input1': input1}, opts)

class MockComponentResource(pulumi.ComponentResource):
    output1: pulumi.Output[str]

    def __init__(self, name: str, opts: Optional[pulumi.ResourceOptions]=None):
        if False:
            print('Hello World!')
        super().__init__('python:test_resource_dep_cycle:MockComponentResource', name, props=None, opts=opts, remote=True)
        self.output1 = self.urn
        self.register_outputs({'output1': self.output1})

@pytest.fixture
def my_mocks():
    if False:
        i = 10
        return i + 15
    old_settings = settings.SETTINGS
    mm = MyMocks()
    mocks.set_mocks(mm, preview=True)
    try:
        yield mm
    finally:
        settings.configure(old_settings)

class MyMocks(pulumi.runtime.Mocks):

    def new_resource(self, args: pulumi.runtime.MockResourceArgs):
        if False:
            for i in range(10):
                print('nop')
        return [args.name + '_id', args.inputs]

    def call(self, args: pulumi.runtime.MockCallArgs):
        if False:
            i = 10
            return i + 15
        return {}
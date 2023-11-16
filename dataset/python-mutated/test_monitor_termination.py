from typing import Any, Optional
import asyncio
import functools
import grpc
import logging
import pytest
import pulumi
from .helpers import raises

@raises(pulumi.RunError)
@pytest.mark.timeout(10)
@pulumi.runtime.test
def test_resource_registration_does_not_hang_when_monitor_unavailable(unavailable_mocks):
    if False:
        return 10
    MyCustom('mycustom', {'inprop': 'hello'})

class Unavailable(grpc.RpcError):

    def code(self):
        if False:
            while True:
                i = 10
        return grpc.StatusCode.UNAVAILABLE

class UnavailableMocks(pulumi.runtime.Mocks):

    def call(self, args: pulumi.runtime.MockCallArgs) -> Any:
        if False:
            i = 10
            return i + 15
        return {}

    def new_resource(self, args: pulumi.runtime.MockResourceArgs) -> Any:
        if False:
            for i in range(10):
                print('nop')
        raise Unavailable()

class MyCustom(pulumi.CustomResource):
    outprop: pulumi.Output[str]

    def __init__(self, resource_name, props: Optional[dict]=None, opts=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__('pkg:index:MyCustom', resource_name, props, opts)
        inprop = (props or {}).get('inprop', None)
        if inprop is None:
            raise TypeError("Missing required property 'inprop'")
        self.outprop = pulumi.Output.from_input(inprop).apply(lambda x: f'output: {x}')

@pytest.fixture
def unavailable_mocks():
    if False:
        for i in range(10):
            print('nop')
    old_settings = pulumi.runtime.settings.SETTINGS
    try:
        mocks = UnavailableMocks()
        pulumi.runtime.mocks.set_mocks(mocks)
        yield mocks
    finally:
        pulumi.runtime.settings.configure(old_settings)
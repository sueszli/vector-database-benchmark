"""Verifies that type-related mistakes in dynamic providers result in
exceptions and not hangs. Regresses
https://github.com/pulumi/pulumi/issues/6981

"""
import contextlib
from typing import Dict
import pytest
import uuid
from pulumi import Input, Output
from pulumi.runtime import settings, mocks
import pulumi
import pulumi.dynamic as dyn
from .helpers import raises

class MyMocks(pulumi.runtime.Mocks):

    def new_resource(self, args: pulumi.runtime.MockResourceArgs):
        if False:
            i = 10
            return i + 15
        result = XProvider().create(args=args.inputs)
        return [result.id, result.outs]

    def call(self, args: pulumi.runtime.MockCallArgs):
        if False:
            return 10
        return {}

@pytest.fixture
def my_mocks():
    if False:
        print('Hello World!')
    old_settings = settings.SETTINGS
    mm = MyMocks()
    mocks.set_mocks(mm, preview=False)
    try:
        yield mm
    finally:
        settings.configure(old_settings)

class XInputs(object):
    x: Input[Dict[str, str]]

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.x = x

class XProvider(dyn.ResourceProvider):

    def create(self, args):
        if False:
            while True:
                i = 10
        outs = {'x': {'my_key_1': {'extra_buggy_key': args['x']['my_key_1'] + '!'}}}
        return dyn.CreateResult(f'schema-{uuid.uuid4()}', outs=outs)

class X(dyn.Resource):
    x: Output[Dict[str, str]]

    def __init__(self, name: str, args: XInputs, opts=None):
        if False:
            while True:
                i = 10
        super().__init__(XProvider(), name, vars(args), opts)

@raises(AssertionError)
@pytest.mark.timeout(10)
@pulumi.runtime.test
def test_pulumi_broken_dynamic_provider(my_mocks):
    if False:
        return 10
    x = X(name='my_x', args=XInputs({'my_key_1': 'my_value_1'}))
    return x.x.apply(print)
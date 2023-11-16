import asyncio
import pytest
import unittest
import pulumi
from pulumi.runtime import mocks
from pulumi import StackReference, StackReferenceOutputDetails

@pytest.mark.asyncio
async def test_stack_reference_output_details(simple_mock):
    ref = StackReference('ref')
    non_secret = await ref.get_output_details('bucket')
    assert StackReferenceOutputDetails(value='mybucket-1234'), non_secret
    secret = await ref.get_output_details('password')
    assert StackReferenceOutputDetails(secret_value='mypassword'), non_secret
    unknown = await ref.get_output_details('does-not-exist')
    assert StackReferenceOutputDetails(), non_secret

@pytest.fixture
def simple_mock():
    if False:
        while True:
            i = 10
    mock = StackReferenceOutputMock()
    mocks.set_mocks(mock)
    yield mock

class StackReferenceOutputMock(pulumi.runtime.Mocks):

    def new_resource(self, args: pulumi.runtime.MockResourceArgs):
        if False:
            i = 10
            return i + 15
        assert 'pulumi:pulumi:StackReference' == args.typ
        return [args.name + '_id', {'outputs': {'bucket': 'mybucket-1234', 'password': pulumi.Output.secret('mypassword')}}]

    def call(self, args: pulumi.runtime.MockCallArgs):
        if False:
            i = 10
            return i + 15
        return {}
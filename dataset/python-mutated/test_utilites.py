import uuid
import warnings
import pytest
from prefect import flow, task
from prefect.client import get_client
from prefect.server import schemas
from prefect.settings import PREFECT_API_DATABASE_CONNECTION_URL
from prefect.testing.utilities import assert_does_not_warn, prefect_test_harness

def test_assert_does_not_warn_no_warning():
    if False:
        print('Hello World!')
    with assert_does_not_warn():
        pass

def test_assert_does_not_warn_does_not_capture_exceptions():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        with assert_does_not_warn():
            raise ValueError()

def test_assert_does_not_warn_raises_assertion_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(AssertionError, match='Warning was raised'):
        with assert_does_not_warn():
            warnings.warn('Test')

@pytest.mark.skip(reason='Test is failing consistently')
async def test_prefect_test_harness():
    very_specific_name = str(uuid.uuid4())

    @task
    def test_task():
        if False:
            i = 10
            return i + 15
        pass

    @flow(name=very_specific_name)
    def test_flow():
        if False:
            while True:
                i = 10
        test_task()
        return 'foo'
    existing_db_url = PREFECT_API_DATABASE_CONNECTION_URL.value()
    with prefect_test_harness():
        async with get_client() as client:
            assert test_flow() == 'foo'
            flows = await client.read_flows(flow_filter=schemas.filters.FlowFilter(name={'any_': [very_specific_name]}))
            assert len(flows) == 1
            assert flows[0].name == very_specific_name
            assert client._ephemeral_app is not None
            assert PREFECT_API_DATABASE_CONNECTION_URL.value() != existing_db_url
    async with get_client() as client:
        flows = await client.read_flows(flow_filter=schemas.filters.FlowFilter(name={'any_': [very_specific_name]}))
        assert len(flows) == 0
    assert PREFECT_API_DATABASE_CONNECTION_URL.value() == existing_db_url
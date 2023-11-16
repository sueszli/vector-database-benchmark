from __future__ import annotations
from typing_extensions import Literal
import pytest
from strawberry.http import GraphQLHTTPResponse
from strawberry.types import ExecutionResult
from .clients.base import HttpClient

def process_result(result: ExecutionResult) -> GraphQLHTTPResponse:
    if False:
        return 10
    if result.data:
        return {'data': {key.upper(): result for (key, result) in result.data.items()}}
    return {}

@pytest.fixture()
def http_client(http_client_class) -> HttpClient:
    if False:
        for i in range(10):
            print('nop')
    return http_client_class(result_override=process_result)

@pytest.mark.parametrize('method', ['get', 'post'])
async def test_custom_process_result(method: Literal['get', 'post'], http_client: HttpClient):
    response = await http_client.query(method=method, query='{ hello }')
    assert response.json['data'] == {'HELLO': 'Hello world'}
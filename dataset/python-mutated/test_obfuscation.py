import pytest
from graphql import GraphQLError
from ....graphql.api import schema
from ..obfuscation import MASK, anonymize_event_payload, anonymize_gql_operation_response, filter_and_hide_headers, obfuscate_url, validate_sensitive_fields_map

@pytest.mark.parametrize(('headers', 'allowed', 'sensitive', 'expected'), [({}, {'header'}, {'header'}, {}), ({'Header': 'val'}, {}, {}, {}), ({'HeAdEr': 'val'}, {'header'}, {}, {'HeAdEr': 'val'}), ({'Header': 'val', 'AuThOrIzAtIoN': 'secret'}, {'header'}, {'authorization'}, {'Header': 'val'}), ({'Content-Length': '10', 'AuThOrIzAtIoN': 'secret', 'Not-Allowed': 'val'}, {'content-length', 'authorization'}, {'authorization'}, {'Content-Length': '10', 'AuThOrIzAtIoN': MASK})])
def test_filter_and_hide_headers(headers, allowed, sensitive, expected):
    if False:
        i = 10
        return i + 15
    assert filter_and_hide_headers(headers, allowed=allowed, sensitive=sensitive) == expected

@pytest.mark.parametrize(('url', 'expected'), [('http://example.com/test', 'http://example.com/test'), ('https://example.com:8000/test/path?q=val&k=val', 'https://example.com:8000/test/path?q=val&k=val'), ('https://user@example.com/test', 'https://user@example.com/test'), ('https://:password@example.com/test', f'https://:{MASK}@example.com/test'), ('http://user:password@example.com:8000/test', f'http://user:{MASK}@example.com:8000/test'), ('awssqs://key:secret@sqs.us-east-2.amazonaws.com/xxxx/myqueue.fifo', f'awssqs://key:{MASK}@sqs.us-east-2.amazonaws.com/xxxx/myqueue.fifo')])
def test_obfuscate_url(url, expected):
    if False:
        i = 10
        return i + 15
    assert obfuscate_url(url) == expected

def test_anonymize_gql_operation_response(gql_operation_factory):
    if False:
        print('Hello World!')
    query = 'query FirstQuery { shop { name } }'
    result = {'data': 'result'}
    sensitive_fields = {'Shop': {'name'}}
    operation_result = gql_operation_factory(query, result=result)
    anonymize_gql_operation_response(operation_result, sensitive_fields)
    assert operation_result.result['data'] == MASK

def test_anonymize_gql_operation_with_mutation_in_query(gql_operation_factory):
    if False:
        while True:
            i = 10
    query = '\n    mutation tokenRefresh($token: String){\n        tokenRefresh(refreshToken: $token){\n            token\n        }\n    }'
    result = {'data': {'tokenRefresh': {'token': 'SECRET TOKEN'}}}
    sensitive_fields = {'RefreshToken': {'token'}}
    operation_result = gql_operation_factory(query, result=result)
    anonymize_gql_operation_response(operation_result, sensitive_fields)
    assert operation_result.result['data'] == MASK

def test_anonymize_gql_operation_with_subscription_in_query(gql_operation_factory):
    if False:
        print('Hello World!')
    query = '\n    subscription{\n      event{\n        ...on ProductUpdated{\n          product{\n            id\n            name\n          }\n        }\n      }\n    }\n    '
    result = {'data': 'secret data'}
    sensitive_fields = {'Product': {'name'}}
    operation_result = gql_operation_factory(query, result=result)
    anonymize_gql_operation_response(operation_result, sensitive_fields)
    assert operation_result.result['data'] == MASK

def test_anonymize_gql_operation_response_with_empty_sensitive_fields_map(gql_operation_factory):
    if False:
        while True:
            i = 10
    query = 'query FirstQuery { shop { name } }'
    result = {'data': 'result'}
    operation_result = gql_operation_factory(query, result=result)
    anonymize_gql_operation_response(operation_result, {})
    assert operation_result.result == result

def test_anonymize_gql_operation_response_with_fragment_spread(gql_operation_factory):
    if False:
        while True:
            i = 10
    query = '\n    fragment ProductFragment on Product {\n      id\n      name\n    }\n    query products($first: Int){\n      products(channel: "channel-pln", first:$first){\n        edges{\n          node{\n            ... ProductFragment\n            variants {\n                variantName: name\n            }\n          }\n        }\n      }\n    }'
    result = {'data': 'result'}
    sensitive_fields = {'Product': {'name'}}
    operation_result = gql_operation_factory(query, result=result)
    anonymize_gql_operation_response(operation_result, sensitive_fields)
    assert operation_result.result['data'] == MASK

@pytest.mark.parametrize('sensitive_fields', [{'NonExistingType': {}}, {'Product': {'nonExistingField'}}, {'Node': {'id'}}])
def test_validate_sensitive_fields_map(sensitive_fields):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(GraphQLError):
        validate_sensitive_fields_map(sensitive_fields, schema)

def test_anonymize_event_payload():
    if False:
        return 10
    query = '\n        subscription{\n          event{\n            ...on ProductUpdated{\n              product{\n                id\n                name\n              }\n            }\n          }\n        }\n        '
    payload = [{'sensitive': 'data'}]
    sensitive_fields = {'Product': {'name'}}
    anonymized = anonymize_event_payload(query, 'any_type', payload, sensitive_fields)
    assert anonymized == MASK

def test_anonymize_event_delivery_payload_when_empty_subscription_query():
    if False:
        while True:
            i = 10
    payload = [{'sensitive': 'data'}]
    sensitive_fields = {'Product': {'name'}}
    anonymized = anonymize_event_payload(None, 'any_type', payload, sensitive_fields)
    assert anonymized == payload
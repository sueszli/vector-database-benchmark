import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
from collections.abc import Iterable
import json
import math
from google.api_core import gapic_v1, grpc_helpers, grpc_helpers_async, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.oauth2 import service_account
from google.protobuf import any_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
from google.rpc import status_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from grafeas.grafeas_v1.services.grafeas import GrafeasAsyncClient, GrafeasClient, pagers, transports
from grafeas.grafeas_v1.types import attestation, build, common, compliance, cvss, deployment, discovery, dsse_attestation, grafeas, image, intoto_provenance, intoto_statement, package, provenance, severity, slsa_provenance, slsa_provenance_zero_two, upgrade, vex, vulnerability

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

@pytest.mark.parametrize('request_type', [grafeas.GetOccurrenceRequest, dict])
def test_get_occurrence(request_type, transport: str='grpc'):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence(name='name_value', resource_uri='resource_uri_value', note_name='note_name_value', kind=common.NoteKind.VULNERABILITY, remediation='remediation_value')
        response = client.get_occurrence(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.GetOccurrenceRequest()
    assert isinstance(response, grafeas.Occurrence)
    assert response.name == 'name_value'
    assert response.resource_uri == 'resource_uri_value'
    assert response.note_name == 'note_name_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.remediation == 'remediation_value'

def test_get_occurrence_empty_call():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_occurrence), '__call__') as call:
        client.get_occurrence()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.GetOccurrenceRequest()

@pytest.mark.asyncio
async def test_get_occurrence_async(transport: str='grpc_asyncio', request_type=grafeas.GetOccurrenceRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_occurrence), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Occurrence(name='name_value', resource_uri='resource_uri_value', note_name='note_name_value', kind=common.NoteKind.VULNERABILITY, remediation='remediation_value'))
        response = await client.get_occurrence(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.GetOccurrenceRequest()
    assert isinstance(response, grafeas.Occurrence)
    assert response.name == 'name_value'
    assert response.resource_uri == 'resource_uri_value'
    assert response.note_name == 'note_name_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.remediation == 'remediation_value'

@pytest.mark.asyncio
async def test_get_occurrence_async_from_dict():
    await test_get_occurrence_async(request_type=dict)

def test_get_occurrence_field_headers():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.GetOccurrenceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence()
        client.get_occurrence(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_occurrence_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.GetOccurrenceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_occurrence), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Occurrence())
        await client.get_occurrence(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_occurrence_flattened():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence()
        client.get_occurrence(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_occurrence_flattened_error():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_occurrence(grafeas.GetOccurrenceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_occurrence_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Occurrence())
        response = await client.get_occurrence(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_occurrence_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_occurrence(grafeas.GetOccurrenceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [grafeas.ListOccurrencesRequest, dict])
def test_list_occurrences(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_occurrences), '__call__') as call:
        call.return_value = grafeas.ListOccurrencesResponse(next_page_token='next_page_token_value')
        response = client.list_occurrences(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.ListOccurrencesRequest()
    assert isinstance(response, pagers.ListOccurrencesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_occurrences_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_occurrences), '__call__') as call:
        client.list_occurrences()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.ListOccurrencesRequest()

@pytest.mark.asyncio
async def test_list_occurrences_async(transport: str='grpc_asyncio', request_type=grafeas.ListOccurrencesRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_occurrences), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.ListOccurrencesResponse(next_page_token='next_page_token_value'))
        response = await client.list_occurrences(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.ListOccurrencesRequest()
    assert isinstance(response, pagers.ListOccurrencesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_occurrences_async_from_dict():
    await test_list_occurrences_async(request_type=dict)

def test_list_occurrences_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.ListOccurrencesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_occurrences), '__call__') as call:
        call.return_value = grafeas.ListOccurrencesResponse()
        client.list_occurrences(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_occurrences_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.ListOccurrencesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_occurrences), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.ListOccurrencesResponse())
        await client.list_occurrences(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_occurrences_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_occurrences), '__call__') as call:
        call.return_value = grafeas.ListOccurrencesResponse()
        client.list_occurrences(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_list_occurrences_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_occurrences(grafeas.ListOccurrencesRequest(), parent='parent_value', filter='filter_value')

@pytest.mark.asyncio
async def test_list_occurrences_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_occurrences), '__call__') as call:
        call.return_value = grafeas.ListOccurrencesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.ListOccurrencesResponse())
        response = await client.list_occurrences(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_occurrences_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_occurrences(grafeas.ListOccurrencesRequest(), parent='parent_value', filter='filter_value')

def test_list_occurrences_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_occurrences), '__call__') as call:
        call.side_effect = (grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_occurrences(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, grafeas.Occurrence) for i in results))

def test_list_occurrences_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_occurrences), '__call__') as call:
        call.side_effect = (grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]), RuntimeError)
        pages = list(client.list_occurrences(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_occurrences_async_pager():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_occurrences), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]), RuntimeError)
        async_pager = await client.list_occurrences(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, grafeas.Occurrence) for i in responses))

@pytest.mark.asyncio
async def test_list_occurrences_async_pages():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_occurrences), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_occurrences(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [grafeas.DeleteOccurrenceRequest, dict])
def test_delete_occurrence(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_occurrence), '__call__') as call:
        call.return_value = None
        response = client.delete_occurrence(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.DeleteOccurrenceRequest()
    assert response is None

def test_delete_occurrence_empty_call():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_occurrence), '__call__') as call:
        client.delete_occurrence()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.DeleteOccurrenceRequest()

@pytest.mark.asyncio
async def test_delete_occurrence_async(transport: str='grpc_asyncio', request_type=grafeas.DeleteOccurrenceRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_occurrence), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_occurrence(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.DeleteOccurrenceRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_occurrence_async_from_dict():
    await test_delete_occurrence_async(request_type=dict)

def test_delete_occurrence_field_headers():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.DeleteOccurrenceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_occurrence), '__call__') as call:
        call.return_value = None
        client.delete_occurrence(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_occurrence_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.DeleteOccurrenceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_occurrence), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_occurrence(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_occurrence_flattened():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_occurrence), '__call__') as call:
        call.return_value = None
        client.delete_occurrence(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_occurrence_flattened_error():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_occurrence(grafeas.DeleteOccurrenceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_occurrence_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_occurrence), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_occurrence(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_occurrence_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_occurrence(grafeas.DeleteOccurrenceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [grafeas.CreateOccurrenceRequest, dict])
def test_create_occurrence(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence(name='name_value', resource_uri='resource_uri_value', note_name='note_name_value', kind=common.NoteKind.VULNERABILITY, remediation='remediation_value')
        response = client.create_occurrence(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.CreateOccurrenceRequest()
    assert isinstance(response, grafeas.Occurrence)
    assert response.name == 'name_value'
    assert response.resource_uri == 'resource_uri_value'
    assert response.note_name == 'note_name_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.remediation == 'remediation_value'

def test_create_occurrence_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_occurrence), '__call__') as call:
        client.create_occurrence()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.CreateOccurrenceRequest()

@pytest.mark.asyncio
async def test_create_occurrence_async(transport: str='grpc_asyncio', request_type=grafeas.CreateOccurrenceRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_occurrence), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Occurrence(name='name_value', resource_uri='resource_uri_value', note_name='note_name_value', kind=common.NoteKind.VULNERABILITY, remediation='remediation_value'))
        response = await client.create_occurrence(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.CreateOccurrenceRequest()
    assert isinstance(response, grafeas.Occurrence)
    assert response.name == 'name_value'
    assert response.resource_uri == 'resource_uri_value'
    assert response.note_name == 'note_name_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.remediation == 'remediation_value'

@pytest.mark.asyncio
async def test_create_occurrence_async_from_dict():
    await test_create_occurrence_async(request_type=dict)

def test_create_occurrence_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.CreateOccurrenceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence()
        client.create_occurrence(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_occurrence_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.CreateOccurrenceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_occurrence), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Occurrence())
        await client.create_occurrence(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_occurrence_flattened():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence()
        client.create_occurrence(parent='parent_value', occurrence=grafeas.Occurrence(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].occurrence
        mock_val = grafeas.Occurrence(name='name_value')
        assert arg == mock_val

def test_create_occurrence_flattened_error():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_occurrence(grafeas.CreateOccurrenceRequest(), parent='parent_value', occurrence=grafeas.Occurrence(name='name_value'))

@pytest.mark.asyncio
async def test_create_occurrence_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Occurrence())
        response = await client.create_occurrence(parent='parent_value', occurrence=grafeas.Occurrence(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].occurrence
        mock_val = grafeas.Occurrence(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_occurrence_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_occurrence(grafeas.CreateOccurrenceRequest(), parent='parent_value', occurrence=grafeas.Occurrence(name='name_value'))

@pytest.mark.parametrize('request_type', [grafeas.BatchCreateOccurrencesRequest, dict])
def test_batch_create_occurrences(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_occurrences), '__call__') as call:
        call.return_value = grafeas.BatchCreateOccurrencesResponse()
        response = client.batch_create_occurrences(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.BatchCreateOccurrencesRequest()
    assert isinstance(response, grafeas.BatchCreateOccurrencesResponse)

def test_batch_create_occurrences_empty_call():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_create_occurrences), '__call__') as call:
        client.batch_create_occurrences()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.BatchCreateOccurrencesRequest()

@pytest.mark.asyncio
async def test_batch_create_occurrences_async(transport: str='grpc_asyncio', request_type=grafeas.BatchCreateOccurrencesRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_occurrences), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.BatchCreateOccurrencesResponse())
        response = await client.batch_create_occurrences(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.BatchCreateOccurrencesRequest()
    assert isinstance(response, grafeas.BatchCreateOccurrencesResponse)

@pytest.mark.asyncio
async def test_batch_create_occurrences_async_from_dict():
    await test_batch_create_occurrences_async(request_type=dict)

def test_batch_create_occurrences_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.BatchCreateOccurrencesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_occurrences), '__call__') as call:
        call.return_value = grafeas.BatchCreateOccurrencesResponse()
        client.batch_create_occurrences(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_create_occurrences_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.BatchCreateOccurrencesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_occurrences), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.BatchCreateOccurrencesResponse())
        await client.batch_create_occurrences(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_create_occurrences_flattened():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_occurrences), '__call__') as call:
        call.return_value = grafeas.BatchCreateOccurrencesResponse()
        client.batch_create_occurrences(parent='parent_value', occurrences=[grafeas.Occurrence(name='name_value')])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].occurrences
        mock_val = [grafeas.Occurrence(name='name_value')]
        assert arg == mock_val

def test_batch_create_occurrences_flattened_error():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_create_occurrences(grafeas.BatchCreateOccurrencesRequest(), parent='parent_value', occurrences=[grafeas.Occurrence(name='name_value')])

@pytest.mark.asyncio
async def test_batch_create_occurrences_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_occurrences), '__call__') as call:
        call.return_value = grafeas.BatchCreateOccurrencesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.BatchCreateOccurrencesResponse())
        response = await client.batch_create_occurrences(parent='parent_value', occurrences=[grafeas.Occurrence(name='name_value')])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].occurrences
        mock_val = [grafeas.Occurrence(name='name_value')]
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_create_occurrences_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_create_occurrences(grafeas.BatchCreateOccurrencesRequest(), parent='parent_value', occurrences=[grafeas.Occurrence(name='name_value')])

@pytest.mark.parametrize('request_type', [grafeas.UpdateOccurrenceRequest, dict])
def test_update_occurrence(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence(name='name_value', resource_uri='resource_uri_value', note_name='note_name_value', kind=common.NoteKind.VULNERABILITY, remediation='remediation_value')
        response = client.update_occurrence(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.UpdateOccurrenceRequest()
    assert isinstance(response, grafeas.Occurrence)
    assert response.name == 'name_value'
    assert response.resource_uri == 'resource_uri_value'
    assert response.note_name == 'note_name_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.remediation == 'remediation_value'

def test_update_occurrence_empty_call():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_occurrence), '__call__') as call:
        client.update_occurrence()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.UpdateOccurrenceRequest()

@pytest.mark.asyncio
async def test_update_occurrence_async(transport: str='grpc_asyncio', request_type=grafeas.UpdateOccurrenceRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_occurrence), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Occurrence(name='name_value', resource_uri='resource_uri_value', note_name='note_name_value', kind=common.NoteKind.VULNERABILITY, remediation='remediation_value'))
        response = await client.update_occurrence(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.UpdateOccurrenceRequest()
    assert isinstance(response, grafeas.Occurrence)
    assert response.name == 'name_value'
    assert response.resource_uri == 'resource_uri_value'
    assert response.note_name == 'note_name_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.remediation == 'remediation_value'

@pytest.mark.asyncio
async def test_update_occurrence_async_from_dict():
    await test_update_occurrence_async(request_type=dict)

def test_update_occurrence_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.UpdateOccurrenceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence()
        client.update_occurrence(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_occurrence_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.UpdateOccurrenceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_occurrence), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Occurrence())
        await client.update_occurrence(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_occurrence_flattened():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence()
        client.update_occurrence(name='name_value', occurrence=grafeas.Occurrence(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].occurrence
        mock_val = grafeas.Occurrence(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_occurrence_flattened_error():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_occurrence(grafeas.UpdateOccurrenceRequest(), name='name_value', occurrence=grafeas.Occurrence(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_occurrence_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_occurrence), '__call__') as call:
        call.return_value = grafeas.Occurrence()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Occurrence())
        response = await client.update_occurrence(name='name_value', occurrence=grafeas.Occurrence(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].occurrence
        mock_val = grafeas.Occurrence(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_occurrence_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_occurrence(grafeas.UpdateOccurrenceRequest(), name='name_value', occurrence=grafeas.Occurrence(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [grafeas.GetOccurrenceNoteRequest, dict])
def test_get_occurrence_note(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_occurrence_note), '__call__') as call:
        call.return_value = grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value'])
        response = client.get_occurrence_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.GetOccurrenceNoteRequest()
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

def test_get_occurrence_note_empty_call():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_occurrence_note), '__call__') as call:
        client.get_occurrence_note()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.GetOccurrenceNoteRequest()

@pytest.mark.asyncio
async def test_get_occurrence_note_async(transport: str='grpc_asyncio', request_type=grafeas.GetOccurrenceNoteRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_occurrence_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value']))
        response = await client.get_occurrence_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.GetOccurrenceNoteRequest()
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

@pytest.mark.asyncio
async def test_get_occurrence_note_async_from_dict():
    await test_get_occurrence_note_async(request_type=dict)

def test_get_occurrence_note_field_headers():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.GetOccurrenceNoteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_occurrence_note), '__call__') as call:
        call.return_value = grafeas.Note()
        client.get_occurrence_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_occurrence_note_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.GetOccurrenceNoteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_occurrence_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note())
        await client.get_occurrence_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_occurrence_note_flattened():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_occurrence_note), '__call__') as call:
        call.return_value = grafeas.Note()
        client.get_occurrence_note(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_occurrence_note_flattened_error():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_occurrence_note(grafeas.GetOccurrenceNoteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_occurrence_note_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_occurrence_note), '__call__') as call:
        call.return_value = grafeas.Note()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note())
        response = await client.get_occurrence_note(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_occurrence_note_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_occurrence_note(grafeas.GetOccurrenceNoteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [grafeas.GetNoteRequest, dict])
def test_get_note(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_note), '__call__') as call:
        call.return_value = grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value'])
        response = client.get_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.GetNoteRequest()
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

def test_get_note_empty_call():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_note), '__call__') as call:
        client.get_note()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.GetNoteRequest()

@pytest.mark.asyncio
async def test_get_note_async(transport: str='grpc_asyncio', request_type=grafeas.GetNoteRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value']))
        response = await client.get_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.GetNoteRequest()
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

@pytest.mark.asyncio
async def test_get_note_async_from_dict():
    await test_get_note_async(request_type=dict)

def test_get_note_field_headers():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.GetNoteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_note), '__call__') as call:
        call.return_value = grafeas.Note()
        client.get_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_note_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.GetNoteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note())
        await client.get_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_note_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_note), '__call__') as call:
        call.return_value = grafeas.Note()
        client.get_note(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_note_flattened_error():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_note(grafeas.GetNoteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_note_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_note), '__call__') as call:
        call.return_value = grafeas.Note()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note())
        response = await client.get_note(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_note_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_note(grafeas.GetNoteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [grafeas.ListNotesRequest, dict])
def test_list_notes(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_notes), '__call__') as call:
        call.return_value = grafeas.ListNotesResponse(next_page_token='next_page_token_value')
        response = client.list_notes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.ListNotesRequest()
    assert isinstance(response, pagers.ListNotesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_notes_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_notes), '__call__') as call:
        client.list_notes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.ListNotesRequest()

@pytest.mark.asyncio
async def test_list_notes_async(transport: str='grpc_asyncio', request_type=grafeas.ListNotesRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_notes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.ListNotesResponse(next_page_token='next_page_token_value'))
        response = await client.list_notes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.ListNotesRequest()
    assert isinstance(response, pagers.ListNotesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_notes_async_from_dict():
    await test_list_notes_async(request_type=dict)

def test_list_notes_field_headers():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.ListNotesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_notes), '__call__') as call:
        call.return_value = grafeas.ListNotesResponse()
        client.list_notes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_notes_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.ListNotesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_notes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.ListNotesResponse())
        await client.list_notes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_notes_flattened():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_notes), '__call__') as call:
        call.return_value = grafeas.ListNotesResponse()
        client.list_notes(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_list_notes_flattened_error():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_notes(grafeas.ListNotesRequest(), parent='parent_value', filter='filter_value')

@pytest.mark.asyncio
async def test_list_notes_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_notes), '__call__') as call:
        call.return_value = grafeas.ListNotesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.ListNotesResponse())
        response = await client.list_notes(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_notes_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_notes(grafeas.ListNotesRequest(), parent='parent_value', filter='filter_value')

def test_list_notes_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_notes), '__call__') as call:
        call.side_effect = (grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note(), grafeas.Note()], next_page_token='abc'), grafeas.ListNotesResponse(notes=[], next_page_token='def'), grafeas.ListNotesResponse(notes=[grafeas.Note()], next_page_token='ghi'), grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_notes(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, grafeas.Note) for i in results))

def test_list_notes_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_notes), '__call__') as call:
        call.side_effect = (grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note(), grafeas.Note()], next_page_token='abc'), grafeas.ListNotesResponse(notes=[], next_page_token='def'), grafeas.ListNotesResponse(notes=[grafeas.Note()], next_page_token='ghi'), grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note()]), RuntimeError)
        pages = list(client.list_notes(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_notes_async_pager():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_notes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note(), grafeas.Note()], next_page_token='abc'), grafeas.ListNotesResponse(notes=[], next_page_token='def'), grafeas.ListNotesResponse(notes=[grafeas.Note()], next_page_token='ghi'), grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note()]), RuntimeError)
        async_pager = await client.list_notes(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, grafeas.Note) for i in responses))

@pytest.mark.asyncio
async def test_list_notes_async_pages():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_notes), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note(), grafeas.Note()], next_page_token='abc'), grafeas.ListNotesResponse(notes=[], next_page_token='def'), grafeas.ListNotesResponse(notes=[grafeas.Note()], next_page_token='ghi'), grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_notes(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [grafeas.DeleteNoteRequest, dict])
def test_delete_note(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_note), '__call__') as call:
        call.return_value = None
        response = client.delete_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.DeleteNoteRequest()
    assert response is None

def test_delete_note_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_note), '__call__') as call:
        client.delete_note()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.DeleteNoteRequest()

@pytest.mark.asyncio
async def test_delete_note_async(transport: str='grpc_asyncio', request_type=grafeas.DeleteNoteRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.DeleteNoteRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_note_async_from_dict():
    await test_delete_note_async(request_type=dict)

def test_delete_note_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.DeleteNoteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_note), '__call__') as call:
        call.return_value = None
        client.delete_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_note_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.DeleteNoteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_note_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_note), '__call__') as call:
        call.return_value = None
        client.delete_note(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_note_flattened_error():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_note(grafeas.DeleteNoteRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_note_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_note), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_note(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_note_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_note(grafeas.DeleteNoteRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [grafeas.CreateNoteRequest, dict])
def test_create_note(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_note), '__call__') as call:
        call.return_value = grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value'])
        response = client.create_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.CreateNoteRequest()
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

def test_create_note_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_note), '__call__') as call:
        client.create_note()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.CreateNoteRequest()

@pytest.mark.asyncio
async def test_create_note_async(transport: str='grpc_asyncio', request_type=grafeas.CreateNoteRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value']))
        response = await client.create_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.CreateNoteRequest()
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

@pytest.mark.asyncio
async def test_create_note_async_from_dict():
    await test_create_note_async(request_type=dict)

def test_create_note_field_headers():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.CreateNoteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_note), '__call__') as call:
        call.return_value = grafeas.Note()
        client.create_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_note_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.CreateNoteRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note())
        await client.create_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_note_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_note), '__call__') as call:
        call.return_value = grafeas.Note()
        client.create_note(parent='parent_value', note_id='note_id_value', note=grafeas.Note(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].note_id
        mock_val = 'note_id_value'
        assert arg == mock_val
        arg = args[0].note
        mock_val = grafeas.Note(name='name_value')
        assert arg == mock_val

def test_create_note_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_note(grafeas.CreateNoteRequest(), parent='parent_value', note_id='note_id_value', note=grafeas.Note(name='name_value'))

@pytest.mark.asyncio
async def test_create_note_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_note), '__call__') as call:
        call.return_value = grafeas.Note()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note())
        response = await client.create_note(parent='parent_value', note_id='note_id_value', note=grafeas.Note(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].note_id
        mock_val = 'note_id_value'
        assert arg == mock_val
        arg = args[0].note
        mock_val = grafeas.Note(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_note_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_note(grafeas.CreateNoteRequest(), parent='parent_value', note_id='note_id_value', note=grafeas.Note(name='name_value'))

@pytest.mark.parametrize('request_type', [grafeas.BatchCreateNotesRequest, dict])
def test_batch_create_notes(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_notes), '__call__') as call:
        call.return_value = grafeas.BatchCreateNotesResponse()
        response = client.batch_create_notes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.BatchCreateNotesRequest()
    assert isinstance(response, grafeas.BatchCreateNotesResponse)

def test_batch_create_notes_empty_call():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_create_notes), '__call__') as call:
        client.batch_create_notes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.BatchCreateNotesRequest()

@pytest.mark.asyncio
async def test_batch_create_notes_async(transport: str='grpc_asyncio', request_type=grafeas.BatchCreateNotesRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_notes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.BatchCreateNotesResponse())
        response = await client.batch_create_notes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.BatchCreateNotesRequest()
    assert isinstance(response, grafeas.BatchCreateNotesResponse)

@pytest.mark.asyncio
async def test_batch_create_notes_async_from_dict():
    await test_batch_create_notes_async(request_type=dict)

def test_batch_create_notes_field_headers():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.BatchCreateNotesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_notes), '__call__') as call:
        call.return_value = grafeas.BatchCreateNotesResponse()
        client.batch_create_notes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_create_notes_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.BatchCreateNotesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_notes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.BatchCreateNotesResponse())
        await client.batch_create_notes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_create_notes_flattened():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_notes), '__call__') as call:
        call.return_value = grafeas.BatchCreateNotesResponse()
        client.batch_create_notes(parent='parent_value', notes={'key_value': grafeas.Note(name='name_value')})
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].notes
        mock_val = {'key_value': grafeas.Note(name='name_value')}
        assert arg == mock_val

def test_batch_create_notes_flattened_error():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_create_notes(grafeas.BatchCreateNotesRequest(), parent='parent_value', notes={'key_value': grafeas.Note(name='name_value')})

@pytest.mark.asyncio
async def test_batch_create_notes_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_notes), '__call__') as call:
        call.return_value = grafeas.BatchCreateNotesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.BatchCreateNotesResponse())
        response = await client.batch_create_notes(parent='parent_value', notes={'key_value': grafeas.Note(name='name_value')})
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].notes
        mock_val = {'key_value': grafeas.Note(name='name_value')}
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_create_notes_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_create_notes(grafeas.BatchCreateNotesRequest(), parent='parent_value', notes={'key_value': grafeas.Note(name='name_value')})

@pytest.mark.parametrize('request_type', [grafeas.UpdateNoteRequest, dict])
def test_update_note(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_note), '__call__') as call:
        call.return_value = grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value'])
        response = client.update_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.UpdateNoteRequest()
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

def test_update_note_empty_call():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_note), '__call__') as call:
        client.update_note()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.UpdateNoteRequest()

@pytest.mark.asyncio
async def test_update_note_async(transport: str='grpc_asyncio', request_type=grafeas.UpdateNoteRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value']))
        response = await client.update_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.UpdateNoteRequest()
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

@pytest.mark.asyncio
async def test_update_note_async_from_dict():
    await test_update_note_async(request_type=dict)

def test_update_note_field_headers():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.UpdateNoteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_note), '__call__') as call:
        call.return_value = grafeas.Note()
        client.update_note(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_note_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.UpdateNoteRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_note), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note())
        await client.update_note(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_note_flattened():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_note), '__call__') as call:
        call.return_value = grafeas.Note()
        client.update_note(name='name_value', note=grafeas.Note(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].note
        mock_val = grafeas.Note(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_note_flattened_error():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_note(grafeas.UpdateNoteRequest(), name='name_value', note=grafeas.Note(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_note_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_note), '__call__') as call:
        call.return_value = grafeas.Note()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.Note())
        response = await client.update_note(name='name_value', note=grafeas.Note(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].note
        mock_val = grafeas.Note(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_note_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_note(grafeas.UpdateNoteRequest(), name='name_value', note=grafeas.Note(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [grafeas.ListNoteOccurrencesRequest, dict])
def test_list_note_occurrences(request_type, transport: str='grpc'):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__') as call:
        call.return_value = grafeas.ListNoteOccurrencesResponse(next_page_token='next_page_token_value')
        response = client.list_note_occurrences(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.ListNoteOccurrencesRequest()
    assert isinstance(response, pagers.ListNoteOccurrencesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_note_occurrences_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__') as call:
        client.list_note_occurrences()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.ListNoteOccurrencesRequest()

@pytest.mark.asyncio
async def test_list_note_occurrences_async(transport: str='grpc_asyncio', request_type=grafeas.ListNoteOccurrencesRequest):
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.ListNoteOccurrencesResponse(next_page_token='next_page_token_value'))
        response = await client.list_note_occurrences(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == grafeas.ListNoteOccurrencesRequest()
    assert isinstance(response, pagers.ListNoteOccurrencesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_note_occurrences_async_from_dict():
    await test_list_note_occurrences_async(request_type=dict)

def test_list_note_occurrences_field_headers():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.ListNoteOccurrencesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__') as call:
        call.return_value = grafeas.ListNoteOccurrencesResponse()
        client.list_note_occurrences(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_note_occurrences_field_headers_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = grafeas.ListNoteOccurrencesRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.ListNoteOccurrencesResponse())
        await client.list_note_occurrences(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_list_note_occurrences_flattened():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__') as call:
        call.return_value = grafeas.ListNoteOccurrencesResponse()
        client.list_note_occurrences(name='name_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_list_note_occurrences_flattened_error():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_note_occurrences(grafeas.ListNoteOccurrencesRequest(), name='name_value', filter='filter_value')

@pytest.mark.asyncio
async def test_list_note_occurrences_flattened_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__') as call:
        call.return_value = grafeas.ListNoteOccurrencesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(grafeas.ListNoteOccurrencesResponse())
        response = await client.list_note_occurrences(name='name_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_note_occurrences_flattened_error_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_note_occurrences(grafeas.ListNoteOccurrencesRequest(), name='name_value', filter='filter_value')

def test_list_note_occurrences_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__') as call:
        call.side_effect = (grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListNoteOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', ''),)),)
        pager = client.list_note_occurrences(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, grafeas.Occurrence) for i in results))

def test_list_note_occurrences_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__') as call:
        call.side_effect = (grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListNoteOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]), RuntimeError)
        pages = list(client.list_note_occurrences(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_note_occurrences_async_pager():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListNoteOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]), RuntimeError)
        async_pager = await client.list_note_occurrences(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, grafeas.Occurrence) for i in responses))

@pytest.mark.asyncio
async def test_list_note_occurrences_async_pages():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_note_occurrences), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListNoteOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_note_occurrences(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [grafeas.GetOccurrenceRequest, dict])
def test_get_occurrence_rest(request_type):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/occurrences/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Occurrence(name='name_value', resource_uri='resource_uri_value', note_name='note_name_value', kind=common.NoteKind.VULNERABILITY, remediation='remediation_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Occurrence.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_occurrence(request)
    assert isinstance(response, grafeas.Occurrence)
    assert response.name == 'name_value'
    assert response.resource_uri == 'resource_uri_value'
    assert response.note_name == 'note_name_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.remediation == 'remediation_value'

def test_get_occurrence_rest_required_fields(request_type=grafeas.GetOccurrenceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_occurrence._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_occurrence._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.Occurrence()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.Occurrence.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_occurrence(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_occurrence_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_occurrence._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_occurrence_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_get_occurrence') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_get_occurrence') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.GetOccurrenceRequest.pb(grafeas.GetOccurrenceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.Occurrence.to_json(grafeas.Occurrence())
        request = grafeas.GetOccurrenceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.Occurrence()
        client.get_occurrence(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_occurrence_rest_bad_request(transport: str='rest', request_type=grafeas.GetOccurrenceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/occurrences/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_occurrence(request)

def test_get_occurrence_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Occurrence()
        sample_request = {'name': 'projects/sample1/occurrences/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Occurrence.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_occurrence(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/occurrences/*}' % client.transport._host, args[1])

def test_get_occurrence_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_occurrence(grafeas.GetOccurrenceRequest(), name='name_value')

def test_get_occurrence_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.ListOccurrencesRequest, dict])
def test_list_occurrences_rest(request_type):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.ListOccurrencesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.ListOccurrencesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_occurrences(request)
    assert isinstance(response, pagers.ListOccurrencesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_occurrences_rest_required_fields(request_type=grafeas.ListOccurrencesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_occurrences._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_occurrences._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.ListOccurrencesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.ListOccurrencesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_occurrences(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_occurrences_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_occurrences._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_occurrences_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_list_occurrences') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_list_occurrences') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.ListOccurrencesRequest.pb(grafeas.ListOccurrencesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.ListOccurrencesResponse.to_json(grafeas.ListOccurrencesResponse())
        request = grafeas.ListOccurrencesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.ListOccurrencesResponse()
        client.list_occurrences(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_occurrences_rest_bad_request(transport: str='rest', request_type=grafeas.ListOccurrencesRequest):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_occurrences(request)

def test_list_occurrences_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.ListOccurrencesResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.ListOccurrencesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_occurrences(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/occurrences' % client.transport._host, args[1])

def test_list_occurrences_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_occurrences(grafeas.ListOccurrencesRequest(), parent='parent_value', filter='filter_value')

def test_list_occurrences_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]))
        response = response + response
        response = tuple((grafeas.ListOccurrencesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_occurrences(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, grafeas.Occurrence) for i in results))
        pages = list(client.list_occurrences(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [grafeas.DeleteOccurrenceRequest, dict])
def test_delete_occurrence_rest(request_type):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/occurrences/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_occurrence(request)
    assert response is None

def test_delete_occurrence_rest_required_fields(request_type=grafeas.DeleteOccurrenceRequest):
    if False:
        return 10
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_occurrence._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_occurrence._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_occurrence(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_occurrence_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_occurrence._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_occurrence_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_delete_occurrence') as pre:
        pre.assert_not_called()
        pb_message = grafeas.DeleteOccurrenceRequest.pb(grafeas.DeleteOccurrenceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = grafeas.DeleteOccurrenceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_occurrence(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_occurrence_rest_bad_request(transport: str='rest', request_type=grafeas.DeleteOccurrenceRequest):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/occurrences/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_occurrence(request)

def test_delete_occurrence_rest_flattened():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/occurrences/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_occurrence(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/occurrences/*}' % client.transport._host, args[1])

def test_delete_occurrence_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_occurrence(grafeas.DeleteOccurrenceRequest(), name='name_value')

def test_delete_occurrence_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.CreateOccurrenceRequest, dict])
def test_create_occurrence_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['occurrence'] = {'name': 'name_value', 'resource_uri': 'resource_uri_value', 'note_name': 'note_name_value', 'kind': 1, 'remediation': 'remediation_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'vulnerability': {'type_': 'type__value', 'severity': 1, 'cvss_score': 0.1082, 'cvssv3': {'base_score': 0.1046, 'exploitability_score': 0.21580000000000002, 'impact_score': 0.1273, 'attack_vector': 1, 'attack_complexity': 1, 'authentication': 1, 'privileges_required': 1, 'user_interaction': 1, 'scope': 1, 'confidentiality_impact': 1, 'integrity_impact': 1, 'availability_impact': 1}, 'package_issue': [{'affected_cpe_uri': 'affected_cpe_uri_value', 'affected_package': 'affected_package_value', 'affected_version': {'epoch': 527, 'name': 'name_value', 'revision': 'revision_value', 'inclusive': True, 'kind': 1, 'full_name': 'full_name_value'}, 'fixed_cpe_uri': 'fixed_cpe_uri_value', 'fixed_package': 'fixed_package_value', 'fixed_version': {}, 'fix_available': True, 'package_type': 'package_type_value', 'effective_severity': 1, 'file_location': [{'file_path': 'file_path_value'}]}], 'short_description': 'short_description_value', 'long_description': 'long_description_value', 'related_urls': [{'url': 'url_value', 'label': 'label_value'}], 'effective_severity': 1, 'fix_available': True, 'cvss_version': 1, 'cvss_v2': {}, 'vex_assessment': {'cve': 'cve_value', 'related_uris': {}, 'note_name': 'note_name_value', 'state': 1, 'impacts': ['impacts_value1', 'impacts_value2'], 'remediations': [{'remediation_type': 1, 'details': 'details_value', 'remediation_uri': {}}], 'justification': {'justification_type': 1, 'details': 'details_value'}}}, 'build': {'provenance': {'id': 'id_value', 'project_id': 'project_id_value', 'commands': [{'name': 'name_value', 'env': ['env_value1', 'env_value2'], 'args': ['args_value1', 'args_value2'], 'dir_': 'dir__value', 'id': 'id_value', 'wait_for': ['wait_for_value1', 'wait_for_value2']}], 'built_artifacts': [{'checksum': 'checksum_value', 'id': 'id_value', 'names': ['names_value1', 'names_value2']}], 'create_time': {}, 'start_time': {}, 'end_time': {}, 'creator': 'creator_value', 'logs_uri': 'logs_uri_value', 'source_provenance': {'artifact_storage_source_uri': 'artifact_storage_source_uri_value', 'file_hashes': {}, 'context': {'cloud_repo': {'repo_id': {'project_repo_id': {'project_id': 'project_id_value', 'repo_name': 'repo_name_value'}, 'uid': 'uid_value'}, 'revision_id': 'revision_id_value', 'alias_context': {'kind': 1, 'name': 'name_value'}}, 'gerrit': {'host_uri': 'host_uri_value', 'gerrit_project': 'gerrit_project_value', 'revision_id': 'revision_id_value', 'alias_context': {}}, 'git': {'url': 'url_value', 'revision_id': 'revision_id_value'}, 'labels': {}}, 'additional_contexts': {}}, 'trigger_id': 'trigger_id_value', 'build_options': {}, 'builder_version': 'builder_version_value'}, 'provenance_bytes': 'provenance_bytes_value', 'intoto_provenance': {'builder_config': {'id': 'id_value'}, 'recipe': {'type_': 'type__value', 'defined_in_material': 1971, 'entry_point': 'entry_point_value', 'arguments': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}], 'environment': {}}, 'metadata': {'build_invocation_id': 'build_invocation_id_value', 'build_started_on': {}, 'build_finished_on': {}, 'completeness': {'arguments': True, 'environment': True, 'materials': True}, 'reproducible': True}, 'materials': ['materials_value1', 'materials_value2']}, 'intoto_statement': {'type_': 'type__value', 'subject': [{'name': 'name_value', 'digest': {}}], 'predicate_type': 'predicate_type_value', 'provenance': {}, 'slsa_provenance': {'builder': {'id': 'id_value'}, 'recipe': {'type_': 'type__value', 'defined_in_material': 1971, 'entry_point': 'entry_point_value', 'arguments': {}, 'environment': {}}, 'metadata': {'build_invocation_id': 'build_invocation_id_value', 'build_started_on': {}, 'build_finished_on': {}, 'completeness': {'arguments': True, 'environment': True, 'materials': True}, 'reproducible': True}, 'materials': [{'uri': 'uri_value', 'digest': {}}]}, 'slsa_provenance_zero_two': {'builder': {'id': 'id_value'}, 'build_type': 'build_type_value', 'invocation': {'config_source': {'uri': 'uri_value', 'digest': {}, 'entry_point': 'entry_point_value'}, 'parameters': {'fields': {}}, 'environment': {}}, 'build_config': {}, 'metadata': {'build_invocation_id': 'build_invocation_id_value', 'build_started_on': {}, 'build_finished_on': {}, 'completeness': {'parameters': True, 'environment': True, 'materials': True}, 'reproducible': True}, 'materials': [{'uri': 'uri_value', 'digest': {}}]}}}, 'image': {'fingerprint': {'v1_name': 'v1_name_value', 'v2_blob': ['v2_blob_value1', 'v2_blob_value2'], 'v2_name': 'v2_name_value'}, 'distance': 843, 'layer_info': [{'directive': 'directive_value', 'arguments': 'arguments_value'}], 'base_resource_url': 'base_resource_url_value'}, 'package': {'name': 'name_value', 'location': [{'cpe_uri': 'cpe_uri_value', 'version': {}, 'path': 'path_value'}], 'package_type': 'package_type_value', 'cpe_uri': 'cpe_uri_value', 'architecture': 1, 'license_': {'expression': 'expression_value', 'comments': 'comments_value'}, 'version': {}}, 'deployment': {'user_email': 'user_email_value', 'deploy_time': {}, 'undeploy_time': {}, 'config': 'config_value', 'address': 'address_value', 'resource_uri': ['resource_uri_value1', 'resource_uri_value2'], 'platform': 1}, 'discovery': {'continuous_analysis': 1, 'analysis_status': 1, 'analysis_completed': {'analysis_type': ['analysis_type_value1', 'analysis_type_value2']}, 'analysis_error': [{'code': 411, 'message': 'message_value', 'details': {}}], 'analysis_status_error': {}, 'cpe': 'cpe_value', 'last_scan_time': {}, 'archive_time': {}}, 'attestation': {'serialized_payload': b'serialized_payload_blob', 'signatures': [{'signature': b'signature_blob', 'public_key_id': 'public_key_id_value'}], 'jwts': [{'compact_jwt': 'compact_jwt_value'}]}, 'upgrade': {'package': 'package_value', 'parsed_version': {}, 'distribution': {'cpe_uri': 'cpe_uri_value', 'classification': 'classification_value', 'severity': 'severity_value', 'cve': ['cve_value1', 'cve_value2']}, 'windows_update': {'identity': {'update_id': 'update_id_value', 'revision': 879}, 'title': 'title_value', 'description': 'description_value', 'categories': [{'category_id': 'category_id_value', 'name': 'name_value'}], 'kb_article_ids': ['kb_article_ids_value1', 'kb_article_ids_value2'], 'support_url': 'support_url_value', 'last_published_timestamp': {}}}, 'compliance': {'non_compliant_files': [{'path': 'path_value', 'display_command': 'display_command_value', 'reason': 'reason_value'}], 'non_compliance_reason': 'non_compliance_reason_value'}, 'dsse_attestation': {'envelope': {'payload': b'payload_blob', 'payload_type': 'payload_type_value', 'signatures': [{'sig': b'sig_blob', 'keyid': 'keyid_value'}]}, 'statement': {}}, 'envelope': {}}
    test_field = grafeas.CreateOccurrenceRequest.meta.fields['occurrence']

    def get_message_fields(field):
        if False:
            return 10
        message_fields = []
        if hasattr(field, 'message') and field.message:
            is_field_type_proto_plus_type = not hasattr(field.message, 'DESCRIPTOR')
            if is_field_type_proto_plus_type:
                message_fields = field.message.meta.fields.values()
            else:
                message_fields = field.message.DESCRIPTOR.fields
        return message_fields
    runtime_nested_fields = [(field.name, nested_field.name) for field in get_message_fields(test_field) for nested_field in get_message_fields(field)]
    subfields_not_in_runtime = []
    for (field, value) in request_init['occurrence'].items():
        result = None
        is_repeated = False
        if isinstance(value, list) and len(value):
            is_repeated = True
            result = value[0]
        if isinstance(value, dict):
            result = value
        if result and hasattr(result, 'keys'):
            for subfield in result.keys():
                if (field, subfield) not in runtime_nested_fields:
                    subfields_not_in_runtime.append({'field': field, 'subfield': subfield, 'is_repeated': is_repeated})
    for subfield_to_delete in subfields_not_in_runtime:
        field = subfield_to_delete.get('field')
        field_repeated = subfield_to_delete.get('is_repeated')
        subfield = subfield_to_delete.get('subfield')
        if subfield:
            if field_repeated:
                for i in range(0, len(request_init['occurrence'][field])):
                    del request_init['occurrence'][field][i][subfield]
            else:
                del request_init['occurrence'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Occurrence(name='name_value', resource_uri='resource_uri_value', note_name='note_name_value', kind=common.NoteKind.VULNERABILITY, remediation='remediation_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Occurrence.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_occurrence(request)
    assert isinstance(response, grafeas.Occurrence)
    assert response.name == 'name_value'
    assert response.resource_uri == 'resource_uri_value'
    assert response.note_name == 'note_name_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.remediation == 'remediation_value'

def test_create_occurrence_rest_required_fields(request_type=grafeas.CreateOccurrenceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_occurrence._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_occurrence._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.Occurrence()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.Occurrence.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_occurrence(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_occurrence_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_occurrence._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'occurrence'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_occurrence_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_create_occurrence') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_create_occurrence') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.CreateOccurrenceRequest.pb(grafeas.CreateOccurrenceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.Occurrence.to_json(grafeas.Occurrence())
        request = grafeas.CreateOccurrenceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.Occurrence()
        client.create_occurrence(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_occurrence_rest_bad_request(transport: str='rest', request_type=grafeas.CreateOccurrenceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_occurrence(request)

def test_create_occurrence_rest_flattened():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Occurrence()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', occurrence=grafeas.Occurrence(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Occurrence.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_occurrence(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/occurrences' % client.transport._host, args[1])

def test_create_occurrence_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_occurrence(grafeas.CreateOccurrenceRequest(), parent='parent_value', occurrence=grafeas.Occurrence(name='name_value'))

def test_create_occurrence_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.BatchCreateOccurrencesRequest, dict])
def test_batch_create_occurrences_rest(request_type):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.BatchCreateOccurrencesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.BatchCreateOccurrencesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_create_occurrences(request)
    assert isinstance(response, grafeas.BatchCreateOccurrencesResponse)

def test_batch_create_occurrences_rest_required_fields(request_type=grafeas.BatchCreateOccurrencesRequest):
    if False:
        return 10
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_occurrences._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_occurrences._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.BatchCreateOccurrencesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.BatchCreateOccurrencesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_create_occurrences(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_create_occurrences_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_create_occurrences._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'occurrences'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_create_occurrences_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_batch_create_occurrences') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_batch_create_occurrences') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.BatchCreateOccurrencesRequest.pb(grafeas.BatchCreateOccurrencesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.BatchCreateOccurrencesResponse.to_json(grafeas.BatchCreateOccurrencesResponse())
        request = grafeas.BatchCreateOccurrencesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.BatchCreateOccurrencesResponse()
        client.batch_create_occurrences(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_create_occurrences_rest_bad_request(transport: str='rest', request_type=grafeas.BatchCreateOccurrencesRequest):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_create_occurrences(request)

def test_batch_create_occurrences_rest_flattened():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.BatchCreateOccurrencesResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', occurrences=[grafeas.Occurrence(name='name_value')])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.BatchCreateOccurrencesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_create_occurrences(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/occurrences:batchCreate' % client.transport._host, args[1])

def test_batch_create_occurrences_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_create_occurrences(grafeas.BatchCreateOccurrencesRequest(), parent='parent_value', occurrences=[grafeas.Occurrence(name='name_value')])

def test_batch_create_occurrences_rest_error():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.UpdateOccurrenceRequest, dict])
def test_update_occurrence_rest(request_type):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/occurrences/sample2'}
    request_init['occurrence'] = {'name': 'name_value', 'resource_uri': 'resource_uri_value', 'note_name': 'note_name_value', 'kind': 1, 'remediation': 'remediation_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'vulnerability': {'type_': 'type__value', 'severity': 1, 'cvss_score': 0.1082, 'cvssv3': {'base_score': 0.1046, 'exploitability_score': 0.21580000000000002, 'impact_score': 0.1273, 'attack_vector': 1, 'attack_complexity': 1, 'authentication': 1, 'privileges_required': 1, 'user_interaction': 1, 'scope': 1, 'confidentiality_impact': 1, 'integrity_impact': 1, 'availability_impact': 1}, 'package_issue': [{'affected_cpe_uri': 'affected_cpe_uri_value', 'affected_package': 'affected_package_value', 'affected_version': {'epoch': 527, 'name': 'name_value', 'revision': 'revision_value', 'inclusive': True, 'kind': 1, 'full_name': 'full_name_value'}, 'fixed_cpe_uri': 'fixed_cpe_uri_value', 'fixed_package': 'fixed_package_value', 'fixed_version': {}, 'fix_available': True, 'package_type': 'package_type_value', 'effective_severity': 1, 'file_location': [{'file_path': 'file_path_value'}]}], 'short_description': 'short_description_value', 'long_description': 'long_description_value', 'related_urls': [{'url': 'url_value', 'label': 'label_value'}], 'effective_severity': 1, 'fix_available': True, 'cvss_version': 1, 'cvss_v2': {}, 'vex_assessment': {'cve': 'cve_value', 'related_uris': {}, 'note_name': 'note_name_value', 'state': 1, 'impacts': ['impacts_value1', 'impacts_value2'], 'remediations': [{'remediation_type': 1, 'details': 'details_value', 'remediation_uri': {}}], 'justification': {'justification_type': 1, 'details': 'details_value'}}}, 'build': {'provenance': {'id': 'id_value', 'project_id': 'project_id_value', 'commands': [{'name': 'name_value', 'env': ['env_value1', 'env_value2'], 'args': ['args_value1', 'args_value2'], 'dir_': 'dir__value', 'id': 'id_value', 'wait_for': ['wait_for_value1', 'wait_for_value2']}], 'built_artifacts': [{'checksum': 'checksum_value', 'id': 'id_value', 'names': ['names_value1', 'names_value2']}], 'create_time': {}, 'start_time': {}, 'end_time': {}, 'creator': 'creator_value', 'logs_uri': 'logs_uri_value', 'source_provenance': {'artifact_storage_source_uri': 'artifact_storage_source_uri_value', 'file_hashes': {}, 'context': {'cloud_repo': {'repo_id': {'project_repo_id': {'project_id': 'project_id_value', 'repo_name': 'repo_name_value'}, 'uid': 'uid_value'}, 'revision_id': 'revision_id_value', 'alias_context': {'kind': 1, 'name': 'name_value'}}, 'gerrit': {'host_uri': 'host_uri_value', 'gerrit_project': 'gerrit_project_value', 'revision_id': 'revision_id_value', 'alias_context': {}}, 'git': {'url': 'url_value', 'revision_id': 'revision_id_value'}, 'labels': {}}, 'additional_contexts': {}}, 'trigger_id': 'trigger_id_value', 'build_options': {}, 'builder_version': 'builder_version_value'}, 'provenance_bytes': 'provenance_bytes_value', 'intoto_provenance': {'builder_config': {'id': 'id_value'}, 'recipe': {'type_': 'type__value', 'defined_in_material': 1971, 'entry_point': 'entry_point_value', 'arguments': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}], 'environment': {}}, 'metadata': {'build_invocation_id': 'build_invocation_id_value', 'build_started_on': {}, 'build_finished_on': {}, 'completeness': {'arguments': True, 'environment': True, 'materials': True}, 'reproducible': True}, 'materials': ['materials_value1', 'materials_value2']}, 'intoto_statement': {'type_': 'type__value', 'subject': [{'name': 'name_value', 'digest': {}}], 'predicate_type': 'predicate_type_value', 'provenance': {}, 'slsa_provenance': {'builder': {'id': 'id_value'}, 'recipe': {'type_': 'type__value', 'defined_in_material': 1971, 'entry_point': 'entry_point_value', 'arguments': {}, 'environment': {}}, 'metadata': {'build_invocation_id': 'build_invocation_id_value', 'build_started_on': {}, 'build_finished_on': {}, 'completeness': {'arguments': True, 'environment': True, 'materials': True}, 'reproducible': True}, 'materials': [{'uri': 'uri_value', 'digest': {}}]}, 'slsa_provenance_zero_two': {'builder': {'id': 'id_value'}, 'build_type': 'build_type_value', 'invocation': {'config_source': {'uri': 'uri_value', 'digest': {}, 'entry_point': 'entry_point_value'}, 'parameters': {'fields': {}}, 'environment': {}}, 'build_config': {}, 'metadata': {'build_invocation_id': 'build_invocation_id_value', 'build_started_on': {}, 'build_finished_on': {}, 'completeness': {'parameters': True, 'environment': True, 'materials': True}, 'reproducible': True}, 'materials': [{'uri': 'uri_value', 'digest': {}}]}}}, 'image': {'fingerprint': {'v1_name': 'v1_name_value', 'v2_blob': ['v2_blob_value1', 'v2_blob_value2'], 'v2_name': 'v2_name_value'}, 'distance': 843, 'layer_info': [{'directive': 'directive_value', 'arguments': 'arguments_value'}], 'base_resource_url': 'base_resource_url_value'}, 'package': {'name': 'name_value', 'location': [{'cpe_uri': 'cpe_uri_value', 'version': {}, 'path': 'path_value'}], 'package_type': 'package_type_value', 'cpe_uri': 'cpe_uri_value', 'architecture': 1, 'license_': {'expression': 'expression_value', 'comments': 'comments_value'}, 'version': {}}, 'deployment': {'user_email': 'user_email_value', 'deploy_time': {}, 'undeploy_time': {}, 'config': 'config_value', 'address': 'address_value', 'resource_uri': ['resource_uri_value1', 'resource_uri_value2'], 'platform': 1}, 'discovery': {'continuous_analysis': 1, 'analysis_status': 1, 'analysis_completed': {'analysis_type': ['analysis_type_value1', 'analysis_type_value2']}, 'analysis_error': [{'code': 411, 'message': 'message_value', 'details': {}}], 'analysis_status_error': {}, 'cpe': 'cpe_value', 'last_scan_time': {}, 'archive_time': {}}, 'attestation': {'serialized_payload': b'serialized_payload_blob', 'signatures': [{'signature': b'signature_blob', 'public_key_id': 'public_key_id_value'}], 'jwts': [{'compact_jwt': 'compact_jwt_value'}]}, 'upgrade': {'package': 'package_value', 'parsed_version': {}, 'distribution': {'cpe_uri': 'cpe_uri_value', 'classification': 'classification_value', 'severity': 'severity_value', 'cve': ['cve_value1', 'cve_value2']}, 'windows_update': {'identity': {'update_id': 'update_id_value', 'revision': 879}, 'title': 'title_value', 'description': 'description_value', 'categories': [{'category_id': 'category_id_value', 'name': 'name_value'}], 'kb_article_ids': ['kb_article_ids_value1', 'kb_article_ids_value2'], 'support_url': 'support_url_value', 'last_published_timestamp': {}}}, 'compliance': {'non_compliant_files': [{'path': 'path_value', 'display_command': 'display_command_value', 'reason': 'reason_value'}], 'non_compliance_reason': 'non_compliance_reason_value'}, 'dsse_attestation': {'envelope': {'payload': b'payload_blob', 'payload_type': 'payload_type_value', 'signatures': [{'sig': b'sig_blob', 'keyid': 'keyid_value'}]}, 'statement': {}}, 'envelope': {}}
    test_field = grafeas.UpdateOccurrenceRequest.meta.fields['occurrence']

    def get_message_fields(field):
        if False:
            for i in range(10):
                print('nop')
        message_fields = []
        if hasattr(field, 'message') and field.message:
            is_field_type_proto_plus_type = not hasattr(field.message, 'DESCRIPTOR')
            if is_field_type_proto_plus_type:
                message_fields = field.message.meta.fields.values()
            else:
                message_fields = field.message.DESCRIPTOR.fields
        return message_fields
    runtime_nested_fields = [(field.name, nested_field.name) for field in get_message_fields(test_field) for nested_field in get_message_fields(field)]
    subfields_not_in_runtime = []
    for (field, value) in request_init['occurrence'].items():
        result = None
        is_repeated = False
        if isinstance(value, list) and len(value):
            is_repeated = True
            result = value[0]
        if isinstance(value, dict):
            result = value
        if result and hasattr(result, 'keys'):
            for subfield in result.keys():
                if (field, subfield) not in runtime_nested_fields:
                    subfields_not_in_runtime.append({'field': field, 'subfield': subfield, 'is_repeated': is_repeated})
    for subfield_to_delete in subfields_not_in_runtime:
        field = subfield_to_delete.get('field')
        field_repeated = subfield_to_delete.get('is_repeated')
        subfield = subfield_to_delete.get('subfield')
        if subfield:
            if field_repeated:
                for i in range(0, len(request_init['occurrence'][field])):
                    del request_init['occurrence'][field][i][subfield]
            else:
                del request_init['occurrence'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Occurrence(name='name_value', resource_uri='resource_uri_value', note_name='note_name_value', kind=common.NoteKind.VULNERABILITY, remediation='remediation_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Occurrence.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_occurrence(request)
    assert isinstance(response, grafeas.Occurrence)
    assert response.name == 'name_value'
    assert response.resource_uri == 'resource_uri_value'
    assert response.note_name == 'note_name_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.remediation == 'remediation_value'

def test_update_occurrence_rest_required_fields(request_type=grafeas.UpdateOccurrenceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_occurrence._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_occurrence._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.Occurrence()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.Occurrence.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_occurrence(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_occurrence_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_occurrence._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('name', 'occurrence'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_occurrence_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_update_occurrence') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_update_occurrence') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.UpdateOccurrenceRequest.pb(grafeas.UpdateOccurrenceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.Occurrence.to_json(grafeas.Occurrence())
        request = grafeas.UpdateOccurrenceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.Occurrence()
        client.update_occurrence(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_occurrence_rest_bad_request(transport: str='rest', request_type=grafeas.UpdateOccurrenceRequest):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/occurrences/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_occurrence(request)

def test_update_occurrence_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Occurrence()
        sample_request = {'name': 'projects/sample1/occurrences/sample2'}
        mock_args = dict(name='name_value', occurrence=grafeas.Occurrence(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Occurrence.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_occurrence(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/occurrences/*}' % client.transport._host, args[1])

def test_update_occurrence_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_occurrence(grafeas.UpdateOccurrenceRequest(), name='name_value', occurrence=grafeas.Occurrence(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_occurrence_rest_error():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.GetOccurrenceNoteRequest, dict])
def test_get_occurrence_note_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/occurrences/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Note.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_occurrence_note(request)
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

def test_get_occurrence_note_rest_required_fields(request_type=grafeas.GetOccurrenceNoteRequest):
    if False:
        return 10
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_occurrence_note._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_occurrence_note._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.Note()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.Note.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_occurrence_note(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_occurrence_note_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_occurrence_note._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_occurrence_note_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_get_occurrence_note') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_get_occurrence_note') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.GetOccurrenceNoteRequest.pb(grafeas.GetOccurrenceNoteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.Note.to_json(grafeas.Note())
        request = grafeas.GetOccurrenceNoteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.Note()
        client.get_occurrence_note(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_occurrence_note_rest_bad_request(transport: str='rest', request_type=grafeas.GetOccurrenceNoteRequest):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/occurrences/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_occurrence_note(request)

def test_get_occurrence_note_rest_flattened():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Note()
        sample_request = {'name': 'projects/sample1/occurrences/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Note.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_occurrence_note(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/occurrences/*}/notes' % client.transport._host, args[1])

def test_get_occurrence_note_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_occurrence_note(grafeas.GetOccurrenceNoteRequest(), name='name_value')

def test_get_occurrence_note_rest_error():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.GetNoteRequest, dict])
def test_get_note_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/notes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Note.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_note(request)
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

def test_get_note_rest_required_fields(request_type=grafeas.GetNoteRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_note._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_note._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.Note()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.Note.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_note(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_note_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_note._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_note_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_get_note') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_get_note') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.GetNoteRequest.pb(grafeas.GetNoteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.Note.to_json(grafeas.Note())
        request = grafeas.GetNoteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.Note()
        client.get_note(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_note_rest_bad_request(transport: str='rest', request_type=grafeas.GetNoteRequest):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/notes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_note(request)

def test_get_note_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Note()
        sample_request = {'name': 'projects/sample1/notes/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Note.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_note(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/notes/*}' % client.transport._host, args[1])

def test_get_note_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_note(grafeas.GetNoteRequest(), name='name_value')

def test_get_note_rest_error():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.ListNotesRequest, dict])
def test_list_notes_rest(request_type):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.ListNotesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.ListNotesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_notes(request)
    assert isinstance(response, pagers.ListNotesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_notes_rest_required_fields(request_type=grafeas.ListNotesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_notes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_notes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.ListNotesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.ListNotesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_notes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_notes_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_notes._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_notes_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_list_notes') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_list_notes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.ListNotesRequest.pb(grafeas.ListNotesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.ListNotesResponse.to_json(grafeas.ListNotesResponse())
        request = grafeas.ListNotesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.ListNotesResponse()
        client.list_notes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_notes_rest_bad_request(transport: str='rest', request_type=grafeas.ListNotesRequest):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_notes(request)

def test_list_notes_rest_flattened():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.ListNotesResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.ListNotesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_notes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/notes' % client.transport._host, args[1])

def test_list_notes_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_notes(grafeas.ListNotesRequest(), parent='parent_value', filter='filter_value')

def test_list_notes_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note(), grafeas.Note()], next_page_token='abc'), grafeas.ListNotesResponse(notes=[], next_page_token='def'), grafeas.ListNotesResponse(notes=[grafeas.Note()], next_page_token='ghi'), grafeas.ListNotesResponse(notes=[grafeas.Note(), grafeas.Note()]))
        response = response + response
        response = tuple((grafeas.ListNotesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_notes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, grafeas.Note) for i in results))
        pages = list(client.list_notes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [grafeas.DeleteNoteRequest, dict])
def test_delete_note_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/notes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_note(request)
    assert response is None

def test_delete_note_rest_required_fields(request_type=grafeas.DeleteNoteRequest):
    if False:
        return 10
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_note._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_note._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_note(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_note_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_note._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_note_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_delete_note') as pre:
        pre.assert_not_called()
        pb_message = grafeas.DeleteNoteRequest.pb(grafeas.DeleteNoteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = grafeas.DeleteNoteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_note(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_note_rest_bad_request(transport: str='rest', request_type=grafeas.DeleteNoteRequest):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/notes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_note(request)

def test_delete_note_rest_flattened():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/notes/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_note(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/notes/*}' % client.transport._host, args[1])

def test_delete_note_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_note(grafeas.DeleteNoteRequest(), name='name_value')

def test_delete_note_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.CreateNoteRequest, dict])
def test_create_note_rest(request_type):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['note'] = {'name': 'name_value', 'short_description': 'short_description_value', 'long_description': 'long_description_value', 'kind': 1, 'related_url': [{'url': 'url_value', 'label': 'label_value'}], 'expiration_time': {'seconds': 751, 'nanos': 543}, 'create_time': {}, 'update_time': {}, 'related_note_names': ['related_note_names_value1', 'related_note_names_value2'], 'vulnerability': {'cvss_score': 0.1082, 'severity': 1, 'details': [{'severity_name': 'severity_name_value', 'description': 'description_value', 'package_type': 'package_type_value', 'affected_cpe_uri': 'affected_cpe_uri_value', 'affected_package': 'affected_package_value', 'affected_version_start': {'epoch': 527, 'name': 'name_value', 'revision': 'revision_value', 'inclusive': True, 'kind': 1, 'full_name': 'full_name_value'}, 'affected_version_end': {}, 'fixed_cpe_uri': 'fixed_cpe_uri_value', 'fixed_package': 'fixed_package_value', 'fixed_version': {}, 'is_obsolete': True, 'source_update_time': {}, 'source': 'source_value', 'vendor': 'vendor_value'}], 'cvss_v3': {'base_score': 0.1046, 'exploitability_score': 0.21580000000000002, 'impact_score': 0.1273, 'attack_vector': 1, 'attack_complexity': 1, 'privileges_required': 1, 'user_interaction': 1, 'scope': 1, 'confidentiality_impact': 1, 'integrity_impact': 1, 'availability_impact': 1}, 'windows_details': [{'cpe_uri': 'cpe_uri_value', 'name': 'name_value', 'description': 'description_value', 'fixing_kbs': [{'name': 'name_value', 'url': 'url_value'}]}], 'source_update_time': {}, 'cvss_version': 1, 'cvss_v2': {'base_score': 0.1046, 'exploitability_score': 0.21580000000000002, 'impact_score': 0.1273, 'attack_vector': 1, 'attack_complexity': 1, 'authentication': 1, 'privileges_required': 1, 'user_interaction': 1, 'scope': 1, 'confidentiality_impact': 1, 'integrity_impact': 1, 'availability_impact': 1}}, 'build': {'builder_version': 'builder_version_value'}, 'image': {'resource_url': 'resource_url_value', 'fingerprint': {'v1_name': 'v1_name_value', 'v2_blob': ['v2_blob_value1', 'v2_blob_value2'], 'v2_name': 'v2_name_value'}}, 'package': {'name': 'name_value', 'distribution': [{'cpe_uri': 'cpe_uri_value', 'architecture': 1, 'latest_version': {}, 'maintainer': 'maintainer_value', 'url': 'url_value', 'description': 'description_value'}], 'package_type': 'package_type_value', 'cpe_uri': 'cpe_uri_value', 'architecture': 1, 'version': {}, 'maintainer': 'maintainer_value', 'url': 'url_value', 'description': 'description_value', 'license_': {'expression': 'expression_value', 'comments': 'comments_value'}, 'digest': [{'algo': 'algo_value', 'digest_bytes': b'digest_bytes_blob'}]}, 'deployment': {'resource_uri': ['resource_uri_value1', 'resource_uri_value2']}, 'discovery': {'analysis_kind': 1}, 'attestation': {'hint': {'human_readable_name': 'human_readable_name_value'}}, 'upgrade': {'package': 'package_value', 'version': {}, 'distributions': [{'cpe_uri': 'cpe_uri_value', 'classification': 'classification_value', 'severity': 'severity_value', 'cve': ['cve_value1', 'cve_value2']}], 'windows_update': {'identity': {'update_id': 'update_id_value', 'revision': 879}, 'title': 'title_value', 'description': 'description_value', 'categories': [{'category_id': 'category_id_value', 'name': 'name_value'}], 'kb_article_ids': ['kb_article_ids_value1', 'kb_article_ids_value2'], 'support_url': 'support_url_value', 'last_published_timestamp': {}}}, 'compliance': {'title': 'title_value', 'description': 'description_value', 'version': [{'cpe_uri': 'cpe_uri_value', 'benchmark_document': 'benchmark_document_value', 'version': 'version_value'}], 'rationale': 'rationale_value', 'remediation': 'remediation_value', 'cis_benchmark': {'profile_level': 1384, 'severity': 1}, 'scan_instructions': b'scan_instructions_blob'}, 'dsse_attestation': {'hint': {'human_readable_name': 'human_readable_name_value'}}, 'vulnerability_assessment': {'title': 'title_value', 'short_description': 'short_description_value', 'long_description': 'long_description_value', 'language_code': 'language_code_value', 'publisher': {'name': 'name_value', 'issuing_authority': 'issuing_authority_value', 'publisher_namespace': 'publisher_namespace_value'}, 'product': {'name': 'name_value', 'id': 'id_value', 'generic_uri': 'generic_uri_value'}, 'assessment': {'cve': 'cve_value', 'short_description': 'short_description_value', 'long_description': 'long_description_value', 'related_uris': {}, 'state': 1, 'impacts': ['impacts_value1', 'impacts_value2'], 'justification': {'justification_type': 1, 'details': 'details_value'}, 'remediations': [{'remediation_type': 1, 'details': 'details_value', 'remediation_uri': {}}]}}}
    test_field = grafeas.CreateNoteRequest.meta.fields['note']

    def get_message_fields(field):
        if False:
            while True:
                i = 10
        message_fields = []
        if hasattr(field, 'message') and field.message:
            is_field_type_proto_plus_type = not hasattr(field.message, 'DESCRIPTOR')
            if is_field_type_proto_plus_type:
                message_fields = field.message.meta.fields.values()
            else:
                message_fields = field.message.DESCRIPTOR.fields
        return message_fields
    runtime_nested_fields = [(field.name, nested_field.name) for field in get_message_fields(test_field) for nested_field in get_message_fields(field)]
    subfields_not_in_runtime = []
    for (field, value) in request_init['note'].items():
        result = None
        is_repeated = False
        if isinstance(value, list) and len(value):
            is_repeated = True
            result = value[0]
        if isinstance(value, dict):
            result = value
        if result and hasattr(result, 'keys'):
            for subfield in result.keys():
                if (field, subfield) not in runtime_nested_fields:
                    subfields_not_in_runtime.append({'field': field, 'subfield': subfield, 'is_repeated': is_repeated})
    for subfield_to_delete in subfields_not_in_runtime:
        field = subfield_to_delete.get('field')
        field_repeated = subfield_to_delete.get('is_repeated')
        subfield = subfield_to_delete.get('subfield')
        if subfield:
            if field_repeated:
                for i in range(0, len(request_init['note'][field])):
                    del request_init['note'][field][i][subfield]
            else:
                del request_init['note'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Note.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_note(request)
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

def test_create_note_rest_required_fields(request_type=grafeas.CreateNoteRequest):
    if False:
        return 10
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['note_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'noteId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_note._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'noteId' in jsonified_request
    assert jsonified_request['noteId'] == request_init['note_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['noteId'] = 'note_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_note._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('note_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'noteId' in jsonified_request
    assert jsonified_request['noteId'] == 'note_id_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.Note()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.Note.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_note(request)
            expected_params = [('noteId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_note_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_note._get_unset_required_fields({})
    assert set(unset_fields) == set(('noteId',)) & set(('parent', 'noteId', 'note'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_note_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_create_note') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_create_note') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.CreateNoteRequest.pb(grafeas.CreateNoteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.Note.to_json(grafeas.Note())
        request = grafeas.CreateNoteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.Note()
        client.create_note(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_note_rest_bad_request(transport: str='rest', request_type=grafeas.CreateNoteRequest):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_note(request)

def test_create_note_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Note()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', note_id='note_id_value', note=grafeas.Note(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Note.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_note(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/notes' % client.transport._host, args[1])

def test_create_note_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_note(grafeas.CreateNoteRequest(), parent='parent_value', note_id='note_id_value', note=grafeas.Note(name='name_value'))

def test_create_note_rest_error():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.BatchCreateNotesRequest, dict])
def test_batch_create_notes_rest(request_type):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.BatchCreateNotesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.BatchCreateNotesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_create_notes(request)
    assert isinstance(response, grafeas.BatchCreateNotesResponse)

def test_batch_create_notes_rest_required_fields(request_type=grafeas.BatchCreateNotesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_notes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_notes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.BatchCreateNotesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.BatchCreateNotesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_create_notes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_create_notes_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_create_notes._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'notes'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_create_notes_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_batch_create_notes') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_batch_create_notes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.BatchCreateNotesRequest.pb(grafeas.BatchCreateNotesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.BatchCreateNotesResponse.to_json(grafeas.BatchCreateNotesResponse())
        request = grafeas.BatchCreateNotesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.BatchCreateNotesResponse()
        client.batch_create_notes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_create_notes_rest_bad_request(transport: str='rest', request_type=grafeas.BatchCreateNotesRequest):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_create_notes(request)

def test_batch_create_notes_rest_flattened():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.BatchCreateNotesResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', notes={'key_value': grafeas.Note(name='name_value')})
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.BatchCreateNotesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_create_notes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/notes:batchCreate' % client.transport._host, args[1])

def test_batch_create_notes_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_create_notes(grafeas.BatchCreateNotesRequest(), parent='parent_value', notes={'key_value': grafeas.Note(name='name_value')})

def test_batch_create_notes_rest_error():
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.UpdateNoteRequest, dict])
def test_update_note_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/notes/sample2'}
    request_init['note'] = {'name': 'name_value', 'short_description': 'short_description_value', 'long_description': 'long_description_value', 'kind': 1, 'related_url': [{'url': 'url_value', 'label': 'label_value'}], 'expiration_time': {'seconds': 751, 'nanos': 543}, 'create_time': {}, 'update_time': {}, 'related_note_names': ['related_note_names_value1', 'related_note_names_value2'], 'vulnerability': {'cvss_score': 0.1082, 'severity': 1, 'details': [{'severity_name': 'severity_name_value', 'description': 'description_value', 'package_type': 'package_type_value', 'affected_cpe_uri': 'affected_cpe_uri_value', 'affected_package': 'affected_package_value', 'affected_version_start': {'epoch': 527, 'name': 'name_value', 'revision': 'revision_value', 'inclusive': True, 'kind': 1, 'full_name': 'full_name_value'}, 'affected_version_end': {}, 'fixed_cpe_uri': 'fixed_cpe_uri_value', 'fixed_package': 'fixed_package_value', 'fixed_version': {}, 'is_obsolete': True, 'source_update_time': {}, 'source': 'source_value', 'vendor': 'vendor_value'}], 'cvss_v3': {'base_score': 0.1046, 'exploitability_score': 0.21580000000000002, 'impact_score': 0.1273, 'attack_vector': 1, 'attack_complexity': 1, 'privileges_required': 1, 'user_interaction': 1, 'scope': 1, 'confidentiality_impact': 1, 'integrity_impact': 1, 'availability_impact': 1}, 'windows_details': [{'cpe_uri': 'cpe_uri_value', 'name': 'name_value', 'description': 'description_value', 'fixing_kbs': [{'name': 'name_value', 'url': 'url_value'}]}], 'source_update_time': {}, 'cvss_version': 1, 'cvss_v2': {'base_score': 0.1046, 'exploitability_score': 0.21580000000000002, 'impact_score': 0.1273, 'attack_vector': 1, 'attack_complexity': 1, 'authentication': 1, 'privileges_required': 1, 'user_interaction': 1, 'scope': 1, 'confidentiality_impact': 1, 'integrity_impact': 1, 'availability_impact': 1}}, 'build': {'builder_version': 'builder_version_value'}, 'image': {'resource_url': 'resource_url_value', 'fingerprint': {'v1_name': 'v1_name_value', 'v2_blob': ['v2_blob_value1', 'v2_blob_value2'], 'v2_name': 'v2_name_value'}}, 'package': {'name': 'name_value', 'distribution': [{'cpe_uri': 'cpe_uri_value', 'architecture': 1, 'latest_version': {}, 'maintainer': 'maintainer_value', 'url': 'url_value', 'description': 'description_value'}], 'package_type': 'package_type_value', 'cpe_uri': 'cpe_uri_value', 'architecture': 1, 'version': {}, 'maintainer': 'maintainer_value', 'url': 'url_value', 'description': 'description_value', 'license_': {'expression': 'expression_value', 'comments': 'comments_value'}, 'digest': [{'algo': 'algo_value', 'digest_bytes': b'digest_bytes_blob'}]}, 'deployment': {'resource_uri': ['resource_uri_value1', 'resource_uri_value2']}, 'discovery': {'analysis_kind': 1}, 'attestation': {'hint': {'human_readable_name': 'human_readable_name_value'}}, 'upgrade': {'package': 'package_value', 'version': {}, 'distributions': [{'cpe_uri': 'cpe_uri_value', 'classification': 'classification_value', 'severity': 'severity_value', 'cve': ['cve_value1', 'cve_value2']}], 'windows_update': {'identity': {'update_id': 'update_id_value', 'revision': 879}, 'title': 'title_value', 'description': 'description_value', 'categories': [{'category_id': 'category_id_value', 'name': 'name_value'}], 'kb_article_ids': ['kb_article_ids_value1', 'kb_article_ids_value2'], 'support_url': 'support_url_value', 'last_published_timestamp': {}}}, 'compliance': {'title': 'title_value', 'description': 'description_value', 'version': [{'cpe_uri': 'cpe_uri_value', 'benchmark_document': 'benchmark_document_value', 'version': 'version_value'}], 'rationale': 'rationale_value', 'remediation': 'remediation_value', 'cis_benchmark': {'profile_level': 1384, 'severity': 1}, 'scan_instructions': b'scan_instructions_blob'}, 'dsse_attestation': {'hint': {'human_readable_name': 'human_readable_name_value'}}, 'vulnerability_assessment': {'title': 'title_value', 'short_description': 'short_description_value', 'long_description': 'long_description_value', 'language_code': 'language_code_value', 'publisher': {'name': 'name_value', 'issuing_authority': 'issuing_authority_value', 'publisher_namespace': 'publisher_namespace_value'}, 'product': {'name': 'name_value', 'id': 'id_value', 'generic_uri': 'generic_uri_value'}, 'assessment': {'cve': 'cve_value', 'short_description': 'short_description_value', 'long_description': 'long_description_value', 'related_uris': {}, 'state': 1, 'impacts': ['impacts_value1', 'impacts_value2'], 'justification': {'justification_type': 1, 'details': 'details_value'}, 'remediations': [{'remediation_type': 1, 'details': 'details_value', 'remediation_uri': {}}]}}}
    test_field = grafeas.UpdateNoteRequest.meta.fields['note']

    def get_message_fields(field):
        if False:
            i = 10
            return i + 15
        message_fields = []
        if hasattr(field, 'message') and field.message:
            is_field_type_proto_plus_type = not hasattr(field.message, 'DESCRIPTOR')
            if is_field_type_proto_plus_type:
                message_fields = field.message.meta.fields.values()
            else:
                message_fields = field.message.DESCRIPTOR.fields
        return message_fields
    runtime_nested_fields = [(field.name, nested_field.name) for field in get_message_fields(test_field) for nested_field in get_message_fields(field)]
    subfields_not_in_runtime = []
    for (field, value) in request_init['note'].items():
        result = None
        is_repeated = False
        if isinstance(value, list) and len(value):
            is_repeated = True
            result = value[0]
        if isinstance(value, dict):
            result = value
        if result and hasattr(result, 'keys'):
            for subfield in result.keys():
                if (field, subfield) not in runtime_nested_fields:
                    subfields_not_in_runtime.append({'field': field, 'subfield': subfield, 'is_repeated': is_repeated})
    for subfield_to_delete in subfields_not_in_runtime:
        field = subfield_to_delete.get('field')
        field_repeated = subfield_to_delete.get('is_repeated')
        subfield = subfield_to_delete.get('subfield')
        if subfield:
            if field_repeated:
                for i in range(0, len(request_init['note'][field])):
                    del request_init['note'][field][i][subfield]
            else:
                del request_init['note'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Note(name='name_value', short_description='short_description_value', long_description='long_description_value', kind=common.NoteKind.VULNERABILITY, related_note_names=['related_note_names_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Note.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_note(request)
    assert isinstance(response, grafeas.Note)
    assert response.name == 'name_value'
    assert response.short_description == 'short_description_value'
    assert response.long_description == 'long_description_value'
    assert response.kind == common.NoteKind.VULNERABILITY
    assert response.related_note_names == ['related_note_names_value']

def test_update_note_rest_required_fields(request_type=grafeas.UpdateNoteRequest):
    if False:
        return 10
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_note._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_note._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.Note()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.Note.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_note(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_note_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_note._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('name', 'note'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_note_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_update_note') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_update_note') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.UpdateNoteRequest.pb(grafeas.UpdateNoteRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.Note.to_json(grafeas.Note())
        request = grafeas.UpdateNoteRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.Note()
        client.update_note(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_note_rest_bad_request(transport: str='rest', request_type=grafeas.UpdateNoteRequest):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/notes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_note(request)

def test_update_note_rest_flattened():
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.Note()
        sample_request = {'name': 'projects/sample1/notes/sample2'}
        mock_args = dict(name='name_value', note=grafeas.Note(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.Note.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_note(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/notes/*}' % client.transport._host, args[1])

def test_update_note_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_note(grafeas.UpdateNoteRequest(), name='name_value', note=grafeas.Note(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_note_rest_error():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [grafeas.ListNoteOccurrencesRequest, dict])
def test_list_note_occurrences_rest(request_type):
    if False:
        return 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/notes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.ListNoteOccurrencesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.ListNoteOccurrencesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_note_occurrences(request)
    assert isinstance(response, pagers.ListNoteOccurrencesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_note_occurrences_rest_required_fields(request_type=grafeas.ListNoteOccurrencesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.GrafeasRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_note_occurrences._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_note_occurrences._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = grafeas.ListNoteOccurrencesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = grafeas.ListNoteOccurrencesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_note_occurrences(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_note_occurrences_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_note_occurrences._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_note_occurrences_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GrafeasRestInterceptor())
    client = GrafeasClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GrafeasRestInterceptor, 'post_list_note_occurrences') as post, mock.patch.object(transports.GrafeasRestInterceptor, 'pre_list_note_occurrences') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = grafeas.ListNoteOccurrencesRequest.pb(grafeas.ListNoteOccurrencesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = grafeas.ListNoteOccurrencesResponse.to_json(grafeas.ListNoteOccurrencesResponse())
        request = grafeas.ListNoteOccurrencesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = grafeas.ListNoteOccurrencesResponse()
        client.list_note_occurrences(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_note_occurrences_rest_bad_request(transport: str='rest', request_type=grafeas.ListNoteOccurrencesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/notes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_note_occurrences(request)

def test_list_note_occurrences_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = grafeas.ListNoteOccurrencesResponse()
        sample_request = {'name': 'projects/sample1/notes/sample2'}
        mock_args = dict(name='name_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = grafeas.ListNoteOccurrencesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_note_occurrences(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/notes/*}/occurrences' % client.transport._host, args[1])

def test_list_note_occurrences_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_note_occurrences(grafeas.ListNoteOccurrencesRequest(), name='name_value', filter='filter_value')

def test_list_note_occurrences_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence(), grafeas.Occurrence()], next_page_token='abc'), grafeas.ListNoteOccurrencesResponse(occurrences=[], next_page_token='def'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence()], next_page_token='ghi'), grafeas.ListNoteOccurrencesResponse(occurrences=[grafeas.Occurrence(), grafeas.Occurrence()]))
        response = response + response
        response = tuple((grafeas.ListNoteOccurrencesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'name': 'projects/sample1/notes/sample2'}
        pager = client.list_note_occurrences(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, grafeas.Occurrence) for i in results))
        pages = list(client.list_note_occurrences(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GrafeasGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = GrafeasClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.GrafeasGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.GrafeasGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.GrafeasGrpcTransport, transports.GrafeasGrpcAsyncIOTransport, transports.GrafeasRestTransport])
def test_transport_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        for i in range(10):
            print('nop')
    transport = GrafeasClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.GrafeasGrpcTransport)

def test_grafeas_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('grafeas.grafeas_v1.services.grafeas.transports.GrafeasTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.GrafeasTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_occurrence', 'list_occurrences', 'delete_occurrence', 'create_occurrence', 'batch_create_occurrences', 'update_occurrence', 'get_occurrence_note', 'get_note', 'list_notes', 'delete_note', 'create_note', 'batch_create_notes', 'update_note', 'list_note_occurrences')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_grafeas_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('grafeas.grafeas_v1.services.grafeas.transports.GrafeasTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.GrafeasTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=(), quota_project_id='octopus')

def test_grafeas_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('grafeas.grafeas_v1.services.grafeas.transports.GrafeasTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.GrafeasTransport()
        adc.assert_called_once()

def test_grafeas_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        GrafeasClient()
        adc.assert_called_once_with(scopes=None, default_scopes=(), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.GrafeasGrpcTransport, transports.GrafeasGrpcAsyncIOTransport])
def test_grafeas_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=(), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.GrafeasGrpcTransport, transports.GrafeasGrpcAsyncIOTransport, transports.GrafeasRestTransport])
def test_grafeas_transport_auth_gdch_credentials(transport_class):
    if False:
        print('Hello World!')
    host = 'https://language.com'
    api_audience_tests = [None, 'https://language2.com']
    api_audience_expect = [host, 'https://language2.com']
    for (t, e) in zip(api_audience_tests, api_audience_expect):
        with mock.patch.object(google.auth, 'default', autospec=True) as adc:
            gdch_mock = mock.MagicMock()
            type(gdch_mock).with_gdch_audience = mock.PropertyMock(return_value=gdch_mock)
            adc.return_value = (gdch_mock, None)
            transport_class(host=host, api_audience=t)
            gdch_mock.with_gdch_audience.assert_called_once_with(e)

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.GrafeasGrpcTransport, grpc_helpers), (transports.GrafeasGrpcAsyncIOTransport, grpc_helpers_async)])
def test_grafeas_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with(':443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=(), scopes=['1', '2'], default_host='', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.GrafeasGrpcTransport, transports.GrafeasGrpcAsyncIOTransport])
def test_grafeas_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch.object(transport_class, 'create_channel') as mock_create_channel:
        mock_ssl_channel_creds = mock.Mock()
        transport_class(host='squid.clam.whelk', credentials=cred, ssl_channel_credentials=mock_ssl_channel_creds)
        mock_create_channel.assert_called_once_with('squid.clam.whelk:443', credentials=cred, credentials_file=None, scopes=None, ssl_credentials=mock_ssl_channel_creds, quota_project_id=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
    with mock.patch.object(transport_class, 'create_channel', return_value=mock.Mock()):
        with mock.patch('grpc.ssl_channel_credentials') as mock_ssl_cred:
            transport_class(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
            (expected_cert, expected_key) = client_cert_source_callback()
            mock_ssl_cred.assert_called_once_with(certificate_chain=expected_cert, private_key=expected_key)

def test_grafeas_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.GrafeasRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_grafeas_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.GrafeasGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_grafeas_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.GrafeasGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.GrafeasGrpcTransport, transports.GrafeasGrpcAsyncIOTransport])
def test_grafeas_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch('grpc.ssl_channel_credentials', autospec=True) as grpc_ssl_channel_cred:
        with mock.patch.object(transport_class, 'create_channel') as grpc_create_channel:
            mock_ssl_cred = mock.Mock()
            grpc_ssl_channel_cred.return_value = mock_ssl_cred
            mock_grpc_channel = mock.Mock()
            grpc_create_channel.return_value = mock_grpc_channel
            cred = ga_credentials.AnonymousCredentials()
            with pytest.warns(DeprecationWarning):
                with mock.patch.object(google.auth, 'default') as adc:
                    adc.return_value = (cred, None)
                    transport = transport_class(host='squid.clam.whelk', api_mtls_endpoint='mtls.squid.clam.whelk', client_cert_source=client_cert_source_callback)
                    adc.assert_called_once()
            grpc_ssl_channel_cred.assert_called_once_with(certificate_chain=b'cert bytes', private_key=b'key bytes')
            grpc_create_channel.assert_called_once_with('mtls.squid.clam.whelk:443', credentials=cred, credentials_file=None, scopes=None, ssl_credentials=mock_ssl_cred, quota_project_id=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
            assert transport.grpc_channel == mock_grpc_channel
            assert transport._ssl_channel_credentials == mock_ssl_cred

@pytest.mark.parametrize('transport_class', [transports.GrafeasGrpcTransport, transports.GrafeasGrpcAsyncIOTransport])
def test_grafeas_transport_channel_mtls_with_adc(transport_class):
    if False:
        i = 10
        return i + 15
    mock_ssl_cred = mock.Mock()
    with mock.patch.multiple('google.auth.transport.grpc.SslCredentials', __init__=mock.Mock(return_value=None), ssl_credentials=mock.PropertyMock(return_value=mock_ssl_cred)):
        with mock.patch.object(transport_class, 'create_channel') as grpc_create_channel:
            mock_grpc_channel = mock.Mock()
            grpc_create_channel.return_value = mock_grpc_channel
            mock_cred = mock.Mock()
            with pytest.warns(DeprecationWarning):
                transport = transport_class(host='squid.clam.whelk', credentials=mock_cred, api_mtls_endpoint='mtls.squid.clam.whelk', client_cert_source=None)
            grpc_create_channel.assert_called_once_with('mtls.squid.clam.whelk:443', credentials=mock_cred, credentials_file=None, scopes=None, ssl_credentials=mock_ssl_cred, quota_project_id=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
            assert transport.grpc_channel == mock_grpc_channel

def test_note_path():
    if False:
        return 10
    project = 'squid'
    note = 'clam'
    expected = 'projects/{project}/notes/{note}'.format(project=project, note=note)
    actual = GrafeasClient.note_path(project, note)
    assert expected == actual

def test_parse_note_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'note': 'octopus'}
    path = GrafeasClient.note_path(**expected)
    actual = GrafeasClient.parse_note_path(path)
    assert expected == actual

def test_occurrence_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    occurrence = 'nudibranch'
    expected = 'projects/{project}/occurrences/{occurrence}'.format(project=project, occurrence=occurrence)
    actual = GrafeasClient.occurrence_path(project, occurrence)
    assert expected == actual

def test_parse_occurrence_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'cuttlefish', 'occurrence': 'mussel'}
    path = GrafeasClient.occurrence_path(**expected)
    actual = GrafeasClient.parse_occurrence_path(path)
    assert expected == actual

def test_project_path():
    if False:
        return 10
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = GrafeasClient.project_path(project)
    assert expected == actual

def test_parse_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus'}
    path = GrafeasClient.project_path(**expected)
    actual = GrafeasClient.parse_project_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = GrafeasClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'abalone'}
    path = GrafeasClient.common_billing_account_path(**expected)
    actual = GrafeasClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = GrafeasClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'clam'}
    path = GrafeasClient.common_folder_path(**expected)
    actual = GrafeasClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = GrafeasClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'octopus'}
    path = GrafeasClient.common_organization_path(**expected)
    actual = GrafeasClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = GrafeasClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch'}
    path = GrafeasClient.common_project_path(**expected)
    actual = GrafeasClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = GrafeasClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = GrafeasClient.common_location_path(**expected)
    actual = GrafeasClient.parse_common_location_path(path)
    assert expected == actual

@pytest.mark.asyncio
async def test_transport_close_async():
    client = GrafeasAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        for i in range(10):
            print('nop')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = GrafeasClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()
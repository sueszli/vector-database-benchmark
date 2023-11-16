import pytest
from azure.core.rest import HttpRequest
from azure.communication.phonenumbers._generated.operations._operations import build_phone_numbers_get_search_result_request, build_phone_numbers_purchase_phone_numbers_request, build_phone_numbers_get_operation_request, build_phone_numbers_cancel_operation_request
test_id = 'test_id'

def test_build_phone_numbers_get_search_result_request():
    if False:
        while True:
            i = 10
    request = build_phone_numbers_get_search_result_request(test_id)
    assert isinstance(request, HttpRequest)
    assert request.method == 'GET'
    assert test_id in request.url
    assert 'api-version=2022-12-01' in request.url
    assert request.headers['Accept'] == 'application/json'

def test_build_phone_numbers_purchase_phone_numbers_request():
    if False:
        while True:
            i = 10
    request = build_phone_numbers_purchase_phone_numbers_request()
    assert isinstance(request, HttpRequest)
    assert request.method == 'POST'
    assert '/availablePhoneNumbers/:purchase' in request.url
    assert request.headers['Accept'] == 'application/json'

def test_build_phone_numbers_get_operation_request():
    if False:
        for i in range(10):
            print('nop')
    request = build_phone_numbers_get_operation_request(test_id)
    assert isinstance(request, HttpRequest)
    assert request.method == 'GET'
    assert test_id in request.url
    assert request.headers['Accept'] == 'application/json'

def test_build_phone_numbers_cancel_operation_request():
    if False:
        print('Hello World!')
    request = build_phone_numbers_cancel_operation_request(test_id)
    assert isinstance(request, HttpRequest)
    assert request.method == 'DELETE'
    assert test_id in request.url
    assert request.headers['Accept'] == 'application/json'
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pytest
import requests_mock
client = Client('api_key', 'api_secret')

def test_invalid_json():
    if False:
        for i in range(10):
            print('nop')
    'Test Invalid response Exception'
    with pytest.raises(BinanceRequestException):
        with requests_mock.mock() as m:
            m.get('https://www.binance.com/exchange-api/v1/public/asset-service/product/get-products', text='<head></html>')
            client.get_products()

def test_api_exception():
    if False:
        return 10
    'Test API response Exception'
    with pytest.raises(BinanceAPIException):
        with requests_mock.mock() as m:
            json_obj = {'code': 1002, 'msg': 'Invalid API call'}
            m.get('https://api.binance.com/api/v3/time', json=json_obj, status_code=400)
            client.get_server_time()

def test_api_exception_invalid_json():
    if False:
        print('Hello World!')
    'Test API response Exception'
    with pytest.raises(BinanceAPIException):
        with requests_mock.mock() as m:
            not_json_str = '<html><body>Error</body></html>'
            m.get('https://api.binance.com/api/v3/time', text=not_json_str, status_code=400)
            client.get_server_time()
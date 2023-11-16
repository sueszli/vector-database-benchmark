import datetime
import math
from typing import Any
import pytest
from superset.utils.hashing import md5_sha_from_dict, md5_sha_from_str

def test_basic_md5_sha():
    if False:
        for i in range(10):
            print('nop')
    obj = {'product': 'Coffee', 'company': 'Gobias Industries', 'price_in_cents': 4000}
    serialized_obj = '{"company": "Gobias Industries", "price_in_cents": 4000, "product": "Coffee"}'
    assert md5_sha_from_str(serialized_obj) == md5_sha_from_dict(obj)
    assert md5_sha_from_str(serialized_obj) == '35f22273cd6a6798b04f8ddef51135e3'

def test_sort_order_md5_sha():
    if False:
        while True:
            i = 10
    obj_1 = {'product': 'Coffee', 'price_in_cents': 4000, 'company': 'Gobias Industries'}
    obj_2 = {'product': 'Coffee', 'company': 'Gobias Industries', 'price_in_cents': 4000}
    assert md5_sha_from_dict(obj_1) == md5_sha_from_dict(obj_2)
    assert md5_sha_from_dict(obj_1) == '35f22273cd6a6798b04f8ddef51135e3'

def test_custom_default_md5_sha():
    if False:
        while True:
            i = 10

    def custom_datetime_serializer(obj: Any):
        if False:
            print('Hello World!')
        if isinstance(obj, datetime.datetime):
            return '<datetime>'
    obj = {'product': 'Coffee', 'company': 'Gobias Industries', 'datetime': datetime.datetime.now()}
    serialized_obj = '{"company": "Gobias Industries", "datetime": "<datetime>", "product": "Coffee"}'
    assert md5_sha_from_str(serialized_obj) == md5_sha_from_dict(obj, default=custom_datetime_serializer)
    assert md5_sha_from_str(serialized_obj) == 'dc280121213aabcaeb8087aef268fd0d'

def test_ignore_nan_md5_sha():
    if False:
        while True:
            i = 10
    obj = {'product': 'Coffee', 'company': 'Gobias Industries', 'price': math.nan}
    serialized_obj = '{"company": "Gobias Industries", "price": NaN, "product": "Coffee"}'
    assert md5_sha_from_str(serialized_obj) == md5_sha_from_dict(obj)
    assert md5_sha_from_str(serialized_obj) == '5d129d1dffebc0bacc734366476d586d'
    serialized_obj = '{"company": "Gobias Industries", "price": null, "product": "Coffee"}'
    assert md5_sha_from_str(serialized_obj) == md5_sha_from_dict(obj, ignore_nan=True)
    assert md5_sha_from_str(serialized_obj) == '40e87d61f6add03816bccdeac5713b9f'
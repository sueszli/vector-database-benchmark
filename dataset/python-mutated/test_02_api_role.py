"""Test Apis."""
import pytest
from .factories import TestApiBase
from .utils import *

@pytest.mark.usefixtures('db')
class TestApiRole(TestApiBase):
    """api role testing"""
    uri_prefix = '/api/role'

    def test_get_list_page_size(self, user, testapp, client):
        if False:
            i = 10
            return i + 15
        'test list should create 2 users at least, due to test pagination, searching.'
        query = {'page': 1, 'size': 1}
        response = {'count': 5}
        resp = client.get('%s/' % self.uri_prefix)
        response_success(resp)
        compare_req_resp(response, resp)
import requests
from airbyte_cdk.sources.declarative.requesters.paginators.no_pagination import NoPagination

def test():
    if False:
        i = 10
        return i + 15
    paginator = NoPagination(parameters={})
    next_page_token = paginator.next_page_token(requests.Response(), [])
    assert next_page_token == {}
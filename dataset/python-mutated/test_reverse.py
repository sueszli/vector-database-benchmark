import pytest
from django.urls import reverse

@pytest.mark.parametrize('view_name, path', [('foobar', '/api/v3/foobar'), ('post_foobar', '/api/v3/foobar'), ('foobar_put', '/api/v3/foobar')])
def test_reverse(view_name, path):
    if False:
        for i in range(10):
            print('nop')
    assert reverse(f'api-3.0.0:{view_name}') == path
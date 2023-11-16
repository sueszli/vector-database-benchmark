from __future__ import annotations
import pytest
from airflow_breeze.utils.provider_dependencies import get_related_providers

def test_get_downstream_only():
    if False:
        return 10
    related_providers = get_related_providers('trino', upstream_dependencies=False, downstream_dependencies=True)
    assert {'openlineage', 'google', 'common.sql'} == related_providers

def test_get_upstream_only():
    if False:
        for i in range(10):
            print('nop')
    related_providers = get_related_providers('trino', upstream_dependencies=True, downstream_dependencies=False)
    assert {'mysql', 'google'} == related_providers

def test_both():
    if False:
        for i in range(10):
            print('nop')
    related_providers = get_related_providers('trino', upstream_dependencies=True, downstream_dependencies=True)
    assert {'openlineage', 'google', 'mysql', 'common.sql'} == related_providers

def test_none():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='.*must be.*'):
        get_related_providers('trino', upstream_dependencies=False, downstream_dependencies=False)
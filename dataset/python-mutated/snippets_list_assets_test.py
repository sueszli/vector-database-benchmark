"""Tests for snippets."""
import os
import pytest
import snippets_list_assets

@pytest.fixture(scope='module')
def organization_id():
    if False:
        for i in range(10):
            print('nop')
    'Get Organization ID from the environment variable'
    return os.environ['GCLOUD_ORGANIZATION']

def test_list_all_assets(organization_id):
    if False:
        i = 10
        return i + 15
    'Demonstrate listing and printing all assets.'
    count = snippets_list_assets.list_all_assets(organization_id)
    assert count > 0

def list_assets_with_filters(organization_id):
    if False:
        while True:
            i = 10
    count = snippets_list_assets.list_all_assets(organization_id)
    assert count > 0

def test_list_assets_with_filters_and_read_time(organization_id):
    if False:
        i = 10
        return i + 15
    count = snippets_list_assets.list_assets_with_filters_and_read_time(organization_id)
    assert count > 0

def test_list_point_in_time_changes(organization_id):
    if False:
        print('Hello World!')
    count = snippets_list_assets.list_point_in_time_changes(organization_id)
    assert count > 0

def test_group_assets(organization_id):
    if False:
        print('Hello World!')
    count = snippets_list_assets.group_assets(organization_id)
    assert count >= 8

def test_group_filtered_assets(organization_id):
    if False:
        i = 10
        return i + 15
    count = snippets_list_assets.group_filtered_assets(organization_id)
    assert count == 0

def test_group_assets_by_changes(organization_id):
    if False:
        i = 10
        return i + 15
    count = snippets_list_assets.group_assets_by_changes(organization_id)
    assert count >= 0
"""Examples for working with organization settings. """
import os
import pytest
import snippets_orgs

@pytest.fixture(scope='module')
def organization_id():
    if False:
        i = 10
        return i + 15
    'Get Organization ID from the environment variable'
    return os.environ['GCLOUD_ORGANIZATION']

def test_get_settings(organization_id):
    if False:
        while True:
            i = 10
    snippets_orgs.get_settings(organization_id)

def test_update_asset_discovery_org_settings(organization_id):
    if False:
        for i in range(10):
            print('nop')
    updated = snippets_orgs.update_asset_discovery_org_settings(organization_id)
    assert updated.enable_asset_discovery
import inspect
import pytest
from azure.mgmt.compute import ComputeManagementClient
AZURE_LOCATION = 'eastus'

class TestMgmtComputeApiVersion:

    def client(self, api_version):
        if False:
            for i in range(10):
                print('nop')
        return ComputeManagementClient(credential='fake_cred', subscription_id='fake_sub_id', api_version=api_version)

    def test_api_version(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            client = self.client(api_version='1000-01-01')
            client.availability_sets.list_by_subscription()
        client = self.client(api_version='2016-04-30-preview')
        signature = inspect.signature(client.availability_sets.list_by_subscription)
        result = 'expand' in signature.parameters.keys()
        try:
            assert result == False
        except AssertionError:
            with pytest.raises(ValueError):
                client.availability_sets.list_by_subscription(expand='fake_expand')
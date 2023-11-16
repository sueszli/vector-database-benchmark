import os
import unittest
import azure.mgmt.databox
from devtools_testutils import AzureMgmtTestCase, ResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtDataBoxTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(MgmtDataBoxTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.databox.DataBoxManagementClient)

    @unittest.skip('unavailable in track2')
    @ResourceGroupPreparer(location=AZURE_LOCATION)
    def test_databox(self, resource_group):
        if False:
            print('Hello World!')
        SUBSCRIPTION_ID = None
        if self.is_live:
            SUBSCRIPTION_ID = os.environ.get('AZURE_SUBSCRIPTION_ID', None)
        if not SUBSCRIPTION_ID:
            SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        RESOURCE_GROUP = resource_group.name
        STORAGE_ACCOUNT_NAME = 'databoxaccountabc'
        JOB_NAME = 'testjob'
        LOCATION_NAME = 'westus'
        BODY = {'details': {'job_details_type': 'DataBox', 'contact_details': {'contact_name': 'Public SDK Test', 'phone': '1234567890', 'phone_extension': '1234', 'email_list': ['testing@microsoft.com']}, 'shipping_address': {'street_address1': '16 TOWNSEND ST', 'street_address2': 'Unit 1', 'city': 'San Francisco', 'state_or_province': 'CA', 'country': 'US', 'postal_code': '94107', 'company_name': 'Microsoft', 'address_type': 'Commercial'}, 'destination_account_details': [{'storage_account_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Storage/storageAccounts/' + STORAGE_ACCOUNT_NAME + '', 'data_destination_type': 'StorageAccount'}]}, 'location': 'westus', 'sku': {'name': 'DataBox'}}
        result = self.mgmt_client.jobs.create(resource_group.name, JOB_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.jobs.get(resource_group.name, JOB_NAME)
        result = self.mgmt_client.jobs.get(resource_group.name, JOB_NAME)
        result = self.mgmt_client.jobs.get(resource_group.name, JOB_NAME)
        result = self.mgmt_client.jobs.get(resource_group.name, JOB_NAME)
        result = self.mgmt_client.jobs.get(resource_group.name, JOB_NAME)
        result = self.mgmt_client.jobs.get(resource_group.name, JOB_NAME)
        result = self.mgmt_client.jobs.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.jobs.list()
        result = self.mgmt_client.operations.list()
        BODY = {'validation_category': 'JobCreationValidation', 'individual_request_details': [{'validation_type': 'ValidateDataDestinationDetails', 'location': 'westus', 'destination_account_details': [{'storage_account_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Storage/storageAccounts/' + STORAGE_ACCOUNT_NAME + '', 'data_destination_type': 'StorageAccount'}]}, {'validation_type': 'ValidateAddress', 'shipping_address': {'street_address1': '16 TOWNSEND ST', 'street_address2': 'Unit 1', 'city': 'San Francisco', 'state_or_province': 'CA', 'country': 'US', 'postal_code': '94107', 'company_name': 'Microsoft', 'address_type': 'Commercial'}, 'device_type': 'DataBox'}]}
        result = self.mgmt_client.service.validate_inputs_by_resource_group(resource_group.name, LOCATION_NAME, BODY)
        BODY = {'country': 'US', 'location': 'westus', 'transfer_type': 'ImportToAzure'}
        result = self.mgmt_client.service.list_available_skus_by_resource_group(resource_group.name, LOCATION_NAME, BODY)
        '\n        # BookShipmentPickupPost[post]\n        now = dt.datetime.now()\n        BODY = {\n          # For new test, change the start time as current date\n          # and end time as start_time + 2 days\n          "start_time": now,\n          "end_time": now + dt.timedelta(days=2),\n          "shipment_location": "Front desk"\n        }\n        self.mgmt_client.jobs.book_shipment_pick_up(resource_group.name, JOB_NAME, BODY)\n        '
        result = self.mgmt_client.jobs.list_credentials(resource_group.name, JOB_NAME)
        BODY = {'details': {'contact_details': {'contact_name': 'Update Job', 'phone': '1234567890', 'phone_extension': '1234', 'email_list': ['testing@microsoft.com']}, 'shipping_address': {'street_address1': '16 TOWNSEND ST', 'street_address2': 'Unit 1', 'city': 'San Francisco', 'state_or_province': 'CA', 'country': 'US', 'postal_code': '94107', 'company_name': 'Microsoft', 'address_type': 'Commercial'}}}
        result = self.mgmt_client.jobs.update(resource_group.name, JOB_NAME, BODY)
        result = result.result()
        BODY = None
        result = self.mgmt_client.service.region_configuration(LOCATION_NAME, BODY)
        BODY = {'validation_type': 'ValidateAddress', 'shipping_address': {'street_address1': '16 TOWNSEND ST', 'street_address2': 'Unit 1', 'city': 'San Francisco', 'state_or_province': 'CA', 'country': 'US', 'postal_code': '94107', 'company_name': 'Microsoft', 'address_type': 'Commercial'}, 'device_type': 'DataBox'}
        result = self.mgmt_client.service.validate_address_method(LOCATION_NAME, BODY)
        BODY = {'validation_category': 'JobCreationValidation', 'individual_request_details': [{'validation_type': 'ValidateDataDestinationDetails', 'location': 'westus', 'destination_account_details': [{'storage_account_id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Storage/storageAccounts/' + STORAGE_ACCOUNT_NAME + '', 'data_destination_type': 'StorageAccount'}]}, {'validation_type': 'ValidateAddress', 'shipping_address': {'street_address1': '16 TOWNSEND ST', 'street_address2': 'Unit 1', 'city': 'San Francisco', 'state_or_province': 'CA', 'country': 'US', 'postal_code': '94107', 'company_name': 'Microsoft', 'address_type': 'Commercial'}, 'device_type': 'DataBox'}]}
        result = self.mgmt_client.service.validate_inputs(LOCATION_NAME, BODY)
        BODY = {'country': 'US', 'location': 'westus', 'transfer_type': 'ImportToAzure'}
        result = self.mgmt_client.service.list_available_skus(LOCATION_NAME, BODY)
        BODY = {'reason': 'CancelTest'}
        result = self.mgmt_client.jobs.cancel(resource_group.name, JOB_NAME, BODY)
        result = self.mgmt_client.jobs.delete(resource_group.name, JOB_NAME)
        result = result.result()
if __name__ == '__main__':
    unittest.main()
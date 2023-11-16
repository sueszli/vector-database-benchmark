from azure.identity import DefaultAzureCredential
from azure.mgmt.databox import DataBoxManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databox\n# USAGE\n    python jobs_create_export.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataBoxManagementClient(credential=DefaultAzureCredential(), subscription_id='YourSubscriptionId')
    response = client.jobs.begin_create(resource_group_name='YourResourceGroupName', job_name='TestJobName1', job_resource={'location': 'westus', 'properties': {'details': {'contactDetails': {'contactName': 'XXXX XXXX', 'emailList': ['xxxx@xxxx.xxx'], 'phone': '0000000000', 'phoneExtension': ''}, 'dataExportDetails': [{'accountDetails': {'dataAccountType': 'StorageAccount', 'storageAccountId': '/subscriptions/YourSubscriptionId/resourceGroups/YourResourceGroupName/providers/Microsoft.Storage/storageAccounts/YourStorageAccountName'}, 'transferConfiguration': {'transferAllDetails': {'include': {'dataAccountType': 'StorageAccount', 'transferAllBlobs': True, 'transferAllFiles': True}}, 'transferConfigurationType': 'TransferAll'}}], 'jobDetailsType': 'DataBox', 'shippingAddress': {'addressType': 'Commercial', 'city': 'XXXX XXXX', 'companyName': 'XXXX XXXX', 'country': 'XX', 'postalCode': '00000', 'stateOrProvince': 'XX', 'streetAddress1': 'XXXX XXXX', 'streetAddress2': 'XXXX XXXX'}}, 'transferType': 'ExportFromAzure'}, 'sku': {'name': 'DataBox'}}).result()
    print(response)
if __name__ == '__main__':
    main()
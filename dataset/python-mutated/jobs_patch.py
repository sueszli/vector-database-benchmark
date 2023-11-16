from azure.identity import DefaultAzureCredential
from azure.mgmt.databox import DataBoxManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databox\n# USAGE\n    python jobs_patch.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = DataBoxManagementClient(credential=DefaultAzureCredential(), subscription_id='YourSubscriptionId')
    response = client.jobs.begin_update(resource_group_name='YourResourceGroupName', job_name='TestJobName1', job_resource_update_parameter={'properties': {'details': {'contactDetails': {'contactName': 'XXXX XXXX', 'emailList': ['xxxx@xxxx.xxx'], 'phone': '0000000000', 'phoneExtension': ''}, 'shippingAddress': {'addressType': 'Commercial', 'city': 'XXXX XXXX', 'companyName': 'XXXX XXXX', 'country': 'XX', 'postalCode': '00000', 'stateOrProvince': 'XX', 'streetAddress1': 'XXXX XXXX', 'streetAddress2': 'XXXX XXXX'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()
from azure.identity import DefaultAzureCredential
from azure.mgmt.databox import DataBoxManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databox\n# USAGE\n    python job_mitigate.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataBoxManagementClient(credential=DefaultAzureCredential(), subscription_id='YourSubscriptionId')
    response = client.mitigate(job_name='TestJobName1', resource_group_name='YourResourceGroupName', mitigate_job_request={'serialNumberCustomerResolutionMap': {'testDISK-1': 'MoveToCleanUpDevice', 'testDISK-2': 'Resume'}})
    print(response)
if __name__ == '__main__':
    main()
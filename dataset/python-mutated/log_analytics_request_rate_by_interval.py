from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python log_analytics_request_rate_by_interval.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.log_analytics.begin_export_request_rate_by_interval(location='westus', parameters={'blobContainerSasUri': 'https://somesasuri', 'fromTime': '2018-01-21T01:54:06.862601Z', 'groupByResourceName': True, 'intervalLength': 'FiveMins', 'toTime': '2018-01-23T01:54:06.862601Z'}).result()
    print(response)
if __name__ == '__main__':
    main()
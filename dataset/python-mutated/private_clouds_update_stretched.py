from azure.identity import DefaultAzureCredential
from azure.mgmt.avs import AVSClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-avs\n# USAGE\n    python private_clouds_update_stretched.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AVSClient(credential=DefaultAzureCredential(), subscription_id='00000000-0000-0000-0000-000000000000')
    response = client.private_clouds.begin_update(resource_group_name='group1', private_cloud_name='cloud1', private_cloud_update={'properties': {'managementCluster': {'clusterSize': 4}}}).result()
    print(response)
if __name__ == '__main__':
    main()
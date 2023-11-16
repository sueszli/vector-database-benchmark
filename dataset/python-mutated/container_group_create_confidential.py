from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerinstance\n# USAGE\n    python container_group_create_confidential.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ContainerInstanceManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.container_groups.begin_create_or_update(resource_group_name='demo', container_group_name='demo1', container_group={'location': 'westeurope', 'properties': {'confidentialComputeProperties': {'ccePolicy': 'eyJhbGxvd19hbGwiOiB0cnVlLCAiY29udGFpbmVycyI6IHsibGVuZ3RoIjogMCwgImVsZW1lbnRzIjogbnVsbH19'}, 'containers': [{'name': 'accdemo', 'properties': {'command': [], 'environmentVariables': [], 'image': 'confiimage', 'ports': [{'port': 8000}], 'resources': {'requests': {'cpu': 1, 'memoryInGB': 1.5}}, 'securityContext': {'capabilities': {'add': ['CAP_NET_ADMIN']}, 'privileged': False}}}], 'imageRegistryCredentials': [], 'ipAddress': {'ports': [{'port': 8000, 'protocol': 'TCP'}], 'type': 'Public'}, 'osType': 'Linux', 'sku': 'Confidential'}}).result()
    print(response)
if __name__ == '__main__':
    main()
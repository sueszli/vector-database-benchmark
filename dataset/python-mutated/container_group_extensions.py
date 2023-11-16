from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerinstance\n# USAGE\n    python container_group_extensions.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ContainerInstanceManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.container_groups.begin_create_or_update(resource_group_name='demo', container_group_name='demo1', container_group={'location': 'eastus2', 'properties': {'containers': [{'name': 'demo1', 'properties': {'command': [], 'environmentVariables': [], 'image': 'nginx', 'ports': [{'port': 80}], 'resources': {'requests': {'cpu': 1, 'memoryInGB': 1.5}}}}], 'extensions': [{'name': 'kube-proxy', 'properties': {'extensionType': 'kube-proxy', 'protectedSettings': {'kubeConfig': '<kubeconfig encoded string>'}, 'settings': {'clusterCidr': '10.240.0.0/16', 'kubeVersion': 'v1.9.10'}, 'version': '1.0'}}, {'name': 'vk-realtime-metrics', 'properties': {'extensionType': 'realtime-metrics', 'version': '1.0'}}], 'imageRegistryCredentials': [], 'ipAddress': {'ports': [{'port': 80, 'protocol': 'TCP'}], 'type': 'Private'}, 'osType': 'Linux', 'subnetIds': [{'id': '/subscriptions/00000000-0000-0000-0000-00000000/resourceGroups/test-rg/providers/Microsoft.Network/virtualNetworks/test-rg-vnet/subnets/test-subnet'}]}}).result()
    print(response)
if __name__ == '__main__':
    main()
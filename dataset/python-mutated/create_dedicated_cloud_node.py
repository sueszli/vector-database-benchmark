from azure.identity import DefaultAzureCredential
from azure.mgmt.vmwarecloudsimple import VMwareCloudSimple
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-vmwarecloudsimple\n# USAGE\n    python create_dedicated_cloud_node.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = VMwareCloudSimple(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.dedicated_cloud_nodes.begin_create_or_update(resource_group_name='myResourceGroup', referer='https://management.azure.com/', dedicated_cloud_node_name='myNode', dedicated_cloud_node_request={'location': 'westus', 'properties': {'availabilityZoneId': 'az1', 'nodesCount': 1, 'placementGroupId': 'n1', 'purchaseId': '56acbd46-3d36-4bbf-9b08-57c30fdf6932', 'skuDescription': {'id': 'general', 'name': 'CS28-Node'}}, 'sku': {'name': 'VMware_CloudSimple_CS28'}}).result()
    print(response)
if __name__ == '__main__':
    main()
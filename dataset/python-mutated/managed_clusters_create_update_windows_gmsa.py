from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservice\n# USAGE\n    python managed_clusters_create_update_windows_gmsa.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ContainerServiceClient(credential=DefaultAzureCredential(), subscription_id='subid1')
    response = client.managed_clusters.begin_create_or_update(resource_group_name='rg1', resource_name='clustername1', parameters={'identity': {'type': 'UserAssigned', 'userAssignedIdentities': {'/subscriptions/subid1/resourceGroups/rgName1/providers/Microsoft.ManagedIdentity/userAssignedIdentities/identity1': {}}}, 'location': 'location1', 'properties': {'addonProfiles': {}, 'agentPoolProfiles': [{'availabilityZones': ['1', '2', '3'], 'count': 3, 'enableNodePublicIP': True, 'mode': 'System', 'name': 'nodepool1', 'osType': 'Linux', 'type': 'VirtualMachineScaleSets', 'vmSize': 'Standard_DS1_v2'}], 'autoScalerProfile': {'scale-down-delay-after-add': '15m', 'scan-interval': '20s'}, 'diskEncryptionSetID': '/subscriptions/subid1/resourceGroups/rg1/providers/Microsoft.Compute/diskEncryptionSets/des', 'dnsPrefix': 'dnsprefix1', 'enablePodSecurityPolicy': True, 'enableRBAC': True, 'kubernetesVersion': '', 'linuxProfile': {'adminUsername': 'azureuser', 'ssh': {'publicKeys': [{'keyData': 'keydata'}]}}, 'networkProfile': {'loadBalancerProfile': {'managedOutboundIPs': {'count': 2}}, 'loadBalancerSku': 'standard', 'outboundType': 'loadBalancer'}, 'servicePrincipalProfile': {'clientId': 'clientid', 'secret': 'secret'}, 'windowsProfile': {'adminPassword': 'replacePassword1234$', 'adminUsername': 'azureuser', 'gmsaProfile': {'enabled': True}}}, 'sku': {'name': 'Basic', 'tier': 'Free'}, 'tags': {'archv2': '', 'tier': 'production'}}).result()
    print(response)
if __name__ == '__main__':
    main()
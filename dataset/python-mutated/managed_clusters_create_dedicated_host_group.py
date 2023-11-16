from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservice\n# USAGE\n    python managed_clusters_create_dedicated_host_group.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = ContainerServiceClient(credential=DefaultAzureCredential(), subscription_id='subid1')
    response = client.managed_clusters.begin_create_or_update(resource_group_name='rg1', resource_name='clustername1', parameters={'location': 'location1', 'properties': {'addonProfiles': {}, 'agentPoolProfiles': [{'count': 3, 'enableNodePublicIP': True, 'hostGroupID': '/subscriptions/subid1/resourcegroups/rg/providers/Microsoft.Compute/hostGroups/hostgroup1', 'name': 'nodepool1', 'osType': 'Linux', 'type': 'VirtualMachineScaleSets', 'vmSize': 'Standard_DS2_v2'}], 'autoScalerProfile': {'scale-down-delay-after-add': '15m', 'scan-interval': '20s'}, 'diskEncryptionSetID': '/subscriptions/subid1/resourceGroups/rg1/providers/Microsoft.Compute/diskEncryptionSets/des', 'dnsPrefix': 'dnsprefix1', 'enablePodSecurityPolicy': False, 'enableRBAC': True, 'kubernetesVersion': '', 'linuxProfile': {'adminUsername': 'azureuser', 'ssh': {'publicKeys': [{'keyData': 'keydata'}]}}, 'networkProfile': {'loadBalancerProfile': {'managedOutboundIPs': {'count': 2}}, 'loadBalancerSku': 'standard', 'outboundType': 'loadBalancer'}, 'servicePrincipalProfile': {'clientId': 'clientid', 'secret': 'secret'}, 'windowsProfile': {'adminPassword': 'replacePassword1234$', 'adminUsername': 'azureuser'}}, 'sku': {'name': 'Basic', 'tier': 'Free'}, 'tags': {'archv2': '', 'tier': 'production'}}).result()
    print(response)
if __name__ == '__main__':
    main()
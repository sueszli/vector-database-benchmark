from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservice\n# USAGE\n    python managed_clusters_create_premium.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = ContainerServiceClient(credential=DefaultAzureCredential(), subscription_id='subid1')
    response = client.managed_clusters.begin_create_or_update(resource_group_name='rg1', resource_name='clustername1', parameters={'location': 'location1', 'properties': {'addonProfiles': {}, 'agentPoolProfiles': [{'count': 3, 'enableEncryptionAtHost': True, 'enableNodePublicIP': True, 'mode': 'System', 'name': 'nodepool1', 'osType': 'Linux', 'type': 'VirtualMachineScaleSets', 'vmSize': 'Standard_DS2_v2'}], 'apiServerAccessProfile': {'disableRunCommand': True}, 'autoScalerProfile': {'scale-down-delay-after-add': '15m', 'scan-interval': '20s'}, 'dnsPrefix': 'dnsprefix1', 'enablePodSecurityPolicy': True, 'enableRBAC': True, 'kubernetesVersion': '', 'linuxProfile': {'adminUsername': 'azureuser', 'ssh': {'publicKeys': [{'keyData': 'keydata'}]}}, 'networkProfile': {'loadBalancerProfile': {'managedOutboundIPs': {'count': 2}}, 'loadBalancerSku': 'standard', 'outboundType': 'loadBalancer'}, 'servicePrincipalProfile': {'clientId': 'clientid', 'secret': 'secret'}, 'supportPlan': 'AKSLongTermSupport', 'windowsProfile': {'adminPassword': 'replacePassword1234$', 'adminUsername': 'azureuser'}}, 'sku': {'name': 'Base', 'tier': 'Premium'}, 'tags': {'archv2': '', 'tier': 'production'}}).result()
    print(response)
if __name__ == '__main__':
    main()
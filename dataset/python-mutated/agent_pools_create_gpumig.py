from azure.identity import DefaultAzureCredential
from azure.mgmt.containerservice import ContainerServiceClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-containerservice\n# USAGE\n    python agent_pools_create_gpumig.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ContainerServiceClient(credential=DefaultAzureCredential(), subscription_id='subid1')
    response = client.agent_pools.begin_create_or_update(resource_group_name='rg1', resource_name='clustername1', agent_pool_name='agentpool1', parameters={'properties': {'count': 3, 'gpuInstanceProfile': 'MIG2g', 'kubeletConfig': {'allowedUnsafeSysctls': ['kernel.msg*', 'net.core.somaxconn'], 'cpuCfsQuota': True, 'cpuCfsQuotaPeriod': '200ms', 'cpuManagerPolicy': 'static', 'failSwapOn': False, 'imageGcHighThreshold': 90, 'imageGcLowThreshold': 70, 'topologyManagerPolicy': 'best-effort'}, 'linuxOSConfig': {'swapFileSizeMB': 1500, 'sysctls': {'kernelThreadsMax': 99999, 'netCoreWmemDefault': 12345, 'netIpv4IpLocalPortRange': '20000 60000', 'netIpv4TcpTwReuse': True}, 'transparentHugePageDefrag': 'madvise', 'transparentHugePageEnabled': 'always'}, 'orchestratorVersion': '', 'osType': 'Linux', 'vmSize': 'Standard_ND96asr_v4'}}).result()
    print(response)
if __name__ == '__main__':
    main()
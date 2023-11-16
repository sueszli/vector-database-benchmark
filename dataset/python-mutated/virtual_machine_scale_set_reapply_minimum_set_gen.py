from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python virtual_machine_scale_set_reapply_minimum_set_gen.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='b4f1213b-cacc-4816-8bfb-f30f90643de8')
    client.virtual_machine_scale_sets.begin_reapply(resource_group_name='VirtualMachineScaleSetReapplyTestRG', vm_scale_set_name='VMSSReapply-Test-ScaleSet').result()
if __name__ == '__main__':
    main()
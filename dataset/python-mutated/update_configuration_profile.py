from azure.identity import DefaultAzureCredential
from azure.mgmt.automanage import AutomanageClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-automanage\n# USAGE\n    python update_configuration_profile.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AutomanageClient(credential=DefaultAzureCredential(), subscription_id='mySubscriptionId')
    response = client.configuration_profiles.update(configuration_profile_name='customConfigurationProfile', resource_group_name='myResourceGroupName', parameters={'properties': {'configuration': {'Antimalware/Enable': False, 'AzureSecurityCenter/Enable': True, 'Backup/Enable': False, 'BootDiagnostics/Enable': True, 'ChangeTrackingAndInventory/Enable': True, 'GuestConfiguration/Enable': True, 'LogAnalytics/Enable': True, 'UpdateManagement/Enable': True, 'VMInsights/Enable': True}}, 'tags': {'Organization': 'Administration'}})
    print(response)
if __name__ == '__main__':
    main()
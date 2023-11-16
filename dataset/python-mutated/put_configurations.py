from azure.identity import DefaultAzureCredential
from azure.mgmt.advisor import AdvisorManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-advisor\n# USAGE\n    python put_configurations.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AdvisorManagementClient(credential=DefaultAzureCredential(), subscription_id='subscriptionId')
    response = client.configurations.create_in_subscription(configuration_name='default', config_contract={'properties': {'digests': [{'actionGroupResourceId': '/subscriptions/subscriptionId/resourceGroups/resourceGroup/providers/microsoft.insights/actionGroups/actionGroupName', 'categories': ['HighAvailability', 'Security', 'Performance', 'Cost', 'OperationalExcellence'], 'frequency': 30, 'language': 'en', 'name': 'digestConfigName', 'state': 'Active'}], 'exclude': True, 'lowCpuThreshold': '5'}})
    print(response)
if __name__ == '__main__':
    main()
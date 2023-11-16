from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python management_group_query.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.query.usage(scope='providers/Microsoft.Management/managementGroups/MyMgId', parameters={'dataset': {'filter': {'and': [{'or': [{'dimensions': {'name': 'ResourceLocation', 'operator': 'In', 'values': ['East US', 'West Europe']}}, {'tags': {'name': 'Environment', 'operator': 'In', 'values': ['UAT', 'Prod']}}]}, {'dimensions': {'name': 'ResourceGroup', 'operator': 'In', 'values': ['API']}}]}, 'granularity': 'Daily'}, 'timeframe': 'MonthToDate', 'type': 'Usage'})
    print(response)
if __name__ == '__main__':
    main()
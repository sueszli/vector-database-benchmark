from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python view_create_or_update_by_resource_group.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.views.create_or_update_by_scope(scope='subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/MYDEVTESTRG', view_name='swaggerExample', parameters={'eTag': '"1d4ff9fe66f1d10"', 'properties': {'accumulated': 'true', 'chart': 'Table', 'displayName': 'swagger Example', 'kpis': [{'enabled': True, 'id': None, 'type': 'Forecast'}, {'enabled': True, 'id': '/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/MYDEVTESTRG/providers/Microsoft.Consumption/budgets/swaggerDemo', 'type': 'Budget'}], 'metric': 'ActualCost', 'pivots': [{'name': 'ServiceName', 'type': 'Dimension'}, {'name': 'MeterCategory', 'type': 'Dimension'}, {'name': 'swaggerTagKey', 'type': 'TagKey'}], 'query': {'dataSet': {'aggregation': {'totalCost': {'function': 'Sum', 'name': 'PreTaxCost'}}, 'granularity': 'Daily', 'grouping': [], 'sorting': [{'direction': 'Ascending', 'name': 'UsageDate'}]}, 'timeframe': 'MonthToDate', 'type': 'Usage'}}})
    print(response)
if __name__ == '__main__':
    main()
from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python external_billing_account_forecast.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.forecast.external_cloud_provider_usage(external_cloud_provider_type='externalBillingAccounts', external_cloud_provider_id='100', parameters={'dataset': {'aggregation': {'totalCost': {'function': 'Sum', 'name': 'Cost'}}, 'filter': {'and': [{'or': [{'dimensions': {'name': 'ResourceLocation', 'operator': 'In', 'values': ['East US', 'West Europe']}}, {'tags': {'name': 'Environment', 'operator': 'In', 'values': ['UAT', 'Prod']}}]}, {'dimensions': {'name': 'ResourceGroup', 'operator': 'In', 'values': ['API']}}]}, 'granularity': 'Daily'}, 'timePeriod': {'from': '2022-08-01T00:00:00+00:00', 'to': '2022-08-31T23:59:59+00:00'}, 'timeframe': 'Custom', 'type': 'Usage'})
    print(response)
if __name__ == '__main__':
    main()
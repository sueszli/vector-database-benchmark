import isodate
from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python log_analytics_get_log_analytics_metrics.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.log_analytics.get_log_analytics_metrics(resource_group_name='RG', profile_name='profile1', metrics=['clientRequestCount'], date_time_begin=isodate.parse_datetime('2020-11-04T04:30:00.000Z'), date_time_end=isodate.parse_datetime('2020-11-04T05:00:00.000Z'), granularity='PT5M', custom_domains=['customdomain1.azurecdn.net', 'customdomain2.azurecdn.net'], protocols=['https'])
    print(response)
if __name__ == '__main__':
    main()
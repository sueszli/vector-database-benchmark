import isodate
from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python log_analytics_get_waf_log_analytics_rankings.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.log_analytics.get_waf_log_analytics_rankings(resource_group_name='RG', profile_name='profile1', metrics=['clientRequestCount'], date_time_begin=isodate.parse_datetime('2020-11-04T06:49:27.554Z'), date_time_end=isodate.parse_datetime('2020-11-04T09:49:27.554Z'), max_ranking='5', rankings=['ruleId'])
    print(response)
if __name__ == '__main__':
    main()
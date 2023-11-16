from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python routes_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.routes.begin_update(resource_group_name='RG', profile_name='profile1', endpoint_name='endpoint1', route_name='route1', route_update_properties={'properties': {'cacheConfiguration': {'compressionSettings': {'contentTypesToCompress': ['text/html', 'application/octet-stream'], 'isCompressionEnabled': True}, 'queryStringCachingBehavior': 'IgnoreQueryString'}, 'customDomains': [{'id': '/subscriptions/subid/resourceGroups/RG/providers/Microsoft.Cdn/profiles/profile1/customDomains/domain1'}], 'enabledState': 'Enabled', 'forwardingProtocol': 'MatchRequest', 'httpsRedirect': 'Enabled', 'linkToDefaultDomain': 'Enabled', 'originGroup': {'id': '/subscriptions/subid/resourceGroups/RG/providers/Microsoft.Cdn/profiles/profile1/originGroups/originGroup1'}, 'originPath': None, 'patternsToMatch': ['/*'], 'ruleSets': [{'id': '/subscriptions/subid/resourceGroups/RG/providers/Microsoft.Cdn/profiles/profile1/ruleSets/ruleSet1'}], 'supportedProtocols': ['Https', 'Http']}}).result()
    print(response)
if __name__ == '__main__':
    main()
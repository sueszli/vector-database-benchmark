from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python security_policies_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.security_policies.begin_create(resource_group_name='RG', profile_name='profile1', security_policy_name='securityPolicy1', security_policy={'properties': {'parameters': {'associations': [{'domains': [{'id': '/subscriptions/subid/resourcegroups/RG/providers/Microsoft.Cdn/profiles/profile1/customdomains/testdomain1'}, {'id': '/subscriptions/subid/resourcegroups/RG/providers/Microsoft.Cdn/profiles/profile1/customdomains/testdomain2'}], 'patternsToMatch': ['/*']}], 'type': 'WebApplicationFirewall', 'wafPolicy': {'id': '/subscriptions/subid/resourcegroups/RG/providers/Microsoft.Network/frontdoorwebapplicationfirewallpolicies/wafTest'}}}}).result()
    print(response)
if __name__ == '__main__':
    main()
from azure.identity import DefaultAzureCredential
from azure.mgmt.cdn import CdnManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-cdn\n# USAGE\n    python rules_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = CdnManagementClient(credential=DefaultAzureCredential(), subscription_id='subid')
    response = client.rules.begin_create(resource_group_name='RG', profile_name='profile1', rule_set_name='ruleSet1', rule_name='rule1', rule={'properties': {'actions': [{'name': 'ModifyResponseHeader', 'parameters': {'headerAction': 'Overwrite', 'headerName': 'X-CDN', 'typeName': 'DeliveryRuleHeaderActionParameters', 'value': 'MSFT'}}], 'conditions': [{'name': 'RequestMethod', 'parameters': {'matchValues': ['GET'], 'negateCondition': False, 'operator': 'Equal', 'typeName': 'DeliveryRuleRequestMethodConditionParameters'}}], 'order': 1}}).result()
    print(response)
if __name__ == '__main__':
    main()
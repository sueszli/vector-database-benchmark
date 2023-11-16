from azure.identity import DefaultAzureCredential
from azure.mgmt.desktopvirtualization import DesktopVirtualizationMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-desktopvirtualization\n# USAGE\n    python application_group_list_by_subscription.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = DesktopVirtualizationMgmtClient(credential=DefaultAzureCredential(), subscription_id='daefabc0-95b4-48b3-b645-8a753a63c4fa')
    response = client.application_groups.list_by_subscription()
    for item in response:
        print(item)
if __name__ == '__main__':
    main()
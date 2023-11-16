from azure.identity import DefaultAzureCredential
from azure.mgmt.agrifood import AgriFoodMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-agrifood\n# USAGE\n    python locations_check_name_availability_available.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AgriFoodMgmtClient(credential=DefaultAzureCredential(), solution_id='SOLUTION_ID', subscription_id='11111111-2222-3333-4444-555555555555')
    response = client.locations.check_name_availability(body={'name': 'newaccountname', 'type': 'Microsoft.AgFoodPlatform/farmBeats'})
    print(response)
if __name__ == '__main__':
    main()
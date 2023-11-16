from azure.identity import DefaultAzureCredential
from azure.mgmt.appcomplianceautomation import AppComplianceAutomationToolForMicrosoft365
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcomplianceautomation\n# USAGE\n    python report_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        return 10
    client = AppComplianceAutomationToolForMicrosoft365(credential=DefaultAzureCredential())
    response = client.report.begin_update(report_name='testReportName', parameters={'properties': {'offerGuid': '0000', 'resources': [{'resourceId': '/subscriptions/00000000-0000-0000-0000-000000000000/resourcegroups/myResourceGroup/providers/Microsoft.Network/privateEndpoints/myPrivateEndpoint', 'tags': {'key1': 'value1'}}], 'timeZone': 'GMT Standard Time', 'triggerTime': '2022-03-04T05:11:56.197Z'}}).result()
    print(response)
if __name__ == '__main__':
    main()
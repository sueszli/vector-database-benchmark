from azure.identity import DefaultAzureCredential
from azure.mgmt.appcomplianceautomation import AppComplianceAutomationToolForMicrosoft365
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcomplianceautomation\n# USAGE\n    python snapshot_download_resource_list.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = AppComplianceAutomationToolForMicrosoft365(credential=DefaultAzureCredential())
    response = client.snapshot.begin_download(report_name='testReportName', snapshot_name='testSnapshotName', parameters={'downloadType': 'ResourceList', 'offerGuid': '00000000-0000-0000-0000-000000000000', 'reportCreatorTenantId': '00000000-0000-0000-0000-000000000000'}).result()
    print(response)
if __name__ == '__main__':
    main()
from azure.identity import DefaultAzureCredential
from azure.mgmt.appcomplianceautomation import AppComplianceAutomationToolForMicrosoft365
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcomplianceautomation\n# USAGE\n    python report_delete.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = AppComplianceAutomationToolForMicrosoft365(credential=DefaultAzureCredential())
    response = client.report.begin_delete(report_name='testReportName').result()
    print(response)
if __name__ == '__main__':
    main()
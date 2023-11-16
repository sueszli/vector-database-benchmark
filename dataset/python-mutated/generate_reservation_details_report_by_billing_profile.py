from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-costmanagement\n# USAGE\n    python generate_reservation_details_report_by_billing_profile.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = CostManagementClient(credential=DefaultAzureCredential())
    response = client.generate_reservation_details_report.begin_by_billing_profile_id(billing_account_id='00000000-0000-0000-0000-000000000000', billing_profile_id='CZSFR-SDFXC-DSDF', start_date='2020-01-01', end_date='2020-01-30').result()
    print(response)
if __name__ == '__main__':
    main()
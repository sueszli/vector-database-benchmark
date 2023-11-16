from azure.identity import DefaultAzureCredential
from azure.mgmt.desktopvirtualization import DesktopVirtualizationMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-desktopvirtualization\n# USAGE\n    python scaling_plan_personal_schedule_get.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = DesktopVirtualizationMgmtClient(credential=DefaultAzureCredential(), subscription_id='daefabc0-95b4-48b3-b645-8a753a63c4fa')
    response = client.scaling_plan_personal_schedules.get(resource_group_name='resourceGroup1', scaling_plan_name='PersonalScalingPlan1', scaling_plan_schedule_name='PersonalScalingPlanSchedule')
    print(response)
if __name__ == '__main__':
    main()
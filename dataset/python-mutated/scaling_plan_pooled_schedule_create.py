from azure.identity import DefaultAzureCredential
from azure.mgmt.desktopvirtualization import DesktopVirtualizationMgmtClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-desktopvirtualization\n# USAGE\n    python scaling_plan_pooled_schedule_create.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = DesktopVirtualizationMgmtClient(credential=DefaultAzureCredential(), subscription_id='daefabc0-95b4-48b3-b645-8a753a63c4fa')
    response = client.scaling_plan_pooled_schedules.create(resource_group_name='resourceGroup1', scaling_plan_name='scalingPlan1', scaling_plan_schedule_name='scalingPlanScheduleWeekdays1', scaling_plan_schedule={'properties': {'daysOfWeek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 'offPeakLoadBalancingAlgorithm': 'DepthFirst', 'offPeakStartTime': {'hour': 20, 'minute': 0}, 'peakLoadBalancingAlgorithm': 'BreadthFirst', 'peakStartTime': {'hour': 8, 'minute': 0}, 'rampDownCapacityThresholdPct': 50, 'rampDownForceLogoffUsers': True, 'rampDownLoadBalancingAlgorithm': 'DepthFirst', 'rampDownMinimumHostsPct': 20, 'rampDownNotificationMessage': 'message', 'rampDownStartTime': {'hour': 18, 'minute': 0}, 'rampDownWaitTimeMinutes': 30, 'rampUpCapacityThresholdPct': 80, 'rampUpLoadBalancingAlgorithm': 'DepthFirst', 'rampUpMinimumHostsPct': 20, 'rampUpStartTime': {'hour': 6, 'minute': 0}}})
    print(response)
if __name__ == '__main__':
    main()
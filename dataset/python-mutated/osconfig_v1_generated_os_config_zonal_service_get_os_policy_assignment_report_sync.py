from google.cloud import osconfig_v1

def sample_get_os_policy_assignment_report():
    if False:
        i = 10
        return i + 15
    client = osconfig_v1.OsConfigZonalServiceClient()
    request = osconfig_v1.GetOSPolicyAssignmentReportRequest(name='name_value')
    response = client.get_os_policy_assignment_report(request=request)
    print(response)
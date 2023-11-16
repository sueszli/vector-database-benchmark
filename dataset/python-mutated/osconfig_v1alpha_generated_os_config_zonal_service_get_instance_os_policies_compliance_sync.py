from google.cloud import osconfig_v1alpha

def sample_get_instance_os_policies_compliance():
    if False:
        i = 10
        return i + 15
    client = osconfig_v1alpha.OsConfigZonalServiceClient()
    request = osconfig_v1alpha.GetInstanceOSPoliciesComplianceRequest(name='name_value')
    response = client.get_instance_os_policies_compliance(request=request)
    print(response)
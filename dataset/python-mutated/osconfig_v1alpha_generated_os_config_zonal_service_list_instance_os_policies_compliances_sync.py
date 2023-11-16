from google.cloud import osconfig_v1alpha

def sample_list_instance_os_policies_compliances():
    if False:
        return 10
    client = osconfig_v1alpha.OsConfigZonalServiceClient()
    request = osconfig_v1alpha.ListInstanceOSPoliciesCompliancesRequest(parent='parent_value')
    page_result = client.list_instance_os_policies_compliances(request=request)
    for response in page_result:
        print(response)
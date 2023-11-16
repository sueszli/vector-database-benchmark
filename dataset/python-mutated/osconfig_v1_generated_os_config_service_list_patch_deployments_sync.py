from google.cloud import osconfig_v1

def sample_list_patch_deployments():
    if False:
        return 10
    client = osconfig_v1.OsConfigServiceClient()
    request = osconfig_v1.ListPatchDeploymentsRequest(parent='parent_value')
    page_result = client.list_patch_deployments(request=request)
    for response in page_result:
        print(response)
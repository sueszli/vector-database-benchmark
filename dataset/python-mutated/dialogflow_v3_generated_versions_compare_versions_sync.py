from google.cloud import dialogflowcx_v3

def sample_compare_versions():
    if False:
        return 10
    client = dialogflowcx_v3.VersionsClient()
    request = dialogflowcx_v3.CompareVersionsRequest(base_version='base_version_value', target_version='target_version_value')
    response = client.compare_versions(request=request)
    print(response)
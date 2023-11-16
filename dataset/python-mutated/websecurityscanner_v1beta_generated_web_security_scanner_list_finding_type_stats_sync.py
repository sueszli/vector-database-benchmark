from google.cloud import websecurityscanner_v1beta

def sample_list_finding_type_stats():
    if False:
        return 10
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.ListFindingTypeStatsRequest(parent='parent_value')
    response = client.list_finding_type_stats(request=request)
    print(response)
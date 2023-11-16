from google.cloud import websecurityscanner_v1alpha

def sample_list_finding_type_stats():
    if False:
        while True:
            i = 10
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    request = websecurityscanner_v1alpha.ListFindingTypeStatsRequest(parent='parent_value')
    response = client.list_finding_type_stats(request=request)
    print(response)
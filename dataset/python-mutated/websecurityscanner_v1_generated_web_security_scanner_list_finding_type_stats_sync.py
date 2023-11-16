from google.cloud import websecurityscanner_v1

def sample_list_finding_type_stats():
    if False:
        print('Hello World!')
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.ListFindingTypeStatsRequest()
    response = client.list_finding_type_stats(request=request)
    print(response)
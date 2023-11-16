from google.cloud import websecurityscanner_v1alpha

def sample_list_scan_runs():
    if False:
        i = 10
        return i + 15
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    request = websecurityscanner_v1alpha.ListScanRunsRequest(parent='parent_value')
    page_result = client.list_scan_runs(request=request)
    for response in page_result:
        print(response)
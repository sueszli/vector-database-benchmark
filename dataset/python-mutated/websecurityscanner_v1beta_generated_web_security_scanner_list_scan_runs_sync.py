from google.cloud import websecurityscanner_v1beta

def sample_list_scan_runs():
    if False:
        while True:
            i = 10
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.ListScanRunsRequest(parent='parent_value')
    page_result = client.list_scan_runs(request=request)
    for response in page_result:
        print(response)
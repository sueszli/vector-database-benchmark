from google.cloud import websecurityscanner_v1

def sample_list_scan_runs():
    if False:
        for i in range(10):
            print('nop')
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.ListScanRunsRequest()
    page_result = client.list_scan_runs(request=request)
    for response in page_result:
        print(response)
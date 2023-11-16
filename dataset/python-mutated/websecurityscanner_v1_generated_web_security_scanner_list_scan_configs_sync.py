from google.cloud import websecurityscanner_v1

def sample_list_scan_configs():
    if False:
        print('Hello World!')
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.ListScanConfigsRequest()
    page_result = client.list_scan_configs(request=request)
    for response in page_result:
        print(response)
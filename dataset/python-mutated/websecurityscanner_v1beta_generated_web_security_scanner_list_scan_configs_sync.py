from google.cloud import websecurityscanner_v1beta

def sample_list_scan_configs():
    if False:
        return 10
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.ListScanConfigsRequest(parent='parent_value')
    page_result = client.list_scan_configs(request=request)
    for response in page_result:
        print(response)
from google.cloud import websecurityscanner_v1alpha

def sample_list_scan_configs():
    if False:
        while True:
            i = 10
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    request = websecurityscanner_v1alpha.ListScanConfigsRequest(parent='parent_value')
    page_result = client.list_scan_configs(request=request)
    for response in page_result:
        print(response)
from google.cloud import websecurityscanner_v1alpha

def sample_create_scan_config():
    if False:
        for i in range(10):
            print('nop')
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    scan_config = websecurityscanner_v1alpha.ScanConfig()
    scan_config.display_name = 'display_name_value'
    scan_config.starting_urls = ['starting_urls_value1', 'starting_urls_value2']
    request = websecurityscanner_v1alpha.CreateScanConfigRequest(parent='parent_value', scan_config=scan_config)
    response = client.create_scan_config(request=request)
    print(response)
from google.cloud import websecurityscanner_v1alpha

def sample_update_scan_config():
    if False:
        while True:
            i = 10
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    scan_config = websecurityscanner_v1alpha.ScanConfig()
    scan_config.display_name = 'display_name_value'
    scan_config.starting_urls = ['starting_urls_value1', 'starting_urls_value2']
    request = websecurityscanner_v1alpha.UpdateScanConfigRequest(scan_config=scan_config)
    response = client.update_scan_config(request=request)
    print(response)
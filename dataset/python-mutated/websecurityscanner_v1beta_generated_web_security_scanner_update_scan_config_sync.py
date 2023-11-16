from google.cloud import websecurityscanner_v1beta

def sample_update_scan_config():
    if False:
        i = 10
        return i + 15
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    scan_config = websecurityscanner_v1beta.ScanConfig()
    scan_config.display_name = 'display_name_value'
    scan_config.starting_urls = ['starting_urls_value1', 'starting_urls_value2']
    request = websecurityscanner_v1beta.UpdateScanConfigRequest(scan_config=scan_config)
    response = client.update_scan_config(request=request)
    print(response)
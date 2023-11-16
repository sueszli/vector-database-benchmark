from google.cloud import websecurityscanner_v1

def sample_delete_scan_config():
    if False:
        return 10
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.DeleteScanConfigRequest()
    client.delete_scan_config(request=request)
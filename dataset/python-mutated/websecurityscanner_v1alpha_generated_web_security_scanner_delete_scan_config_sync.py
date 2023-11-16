from google.cloud import websecurityscanner_v1alpha

def sample_delete_scan_config():
    if False:
        return 10
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    request = websecurityscanner_v1alpha.DeleteScanConfigRequest(name='name_value')
    client.delete_scan_config(request=request)
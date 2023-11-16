from google.cloud import websecurityscanner_v1

def sample_update_scan_config():
    if False:
        print('Hello World!')
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.UpdateScanConfigRequest()
    response = client.update_scan_config(request=request)
    print(response)
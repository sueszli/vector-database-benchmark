from google.cloud import websecurityscanner_v1

def sample_get_scan_config():
    if False:
        i = 10
        return i + 15
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.GetScanConfigRequest()
    response = client.get_scan_config(request=request)
    print(response)
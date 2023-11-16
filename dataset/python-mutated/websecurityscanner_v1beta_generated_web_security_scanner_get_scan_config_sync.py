from google.cloud import websecurityscanner_v1beta

def sample_get_scan_config():
    if False:
        while True:
            i = 10
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.GetScanConfigRequest(name='name_value')
    response = client.get_scan_config(request=request)
    print(response)
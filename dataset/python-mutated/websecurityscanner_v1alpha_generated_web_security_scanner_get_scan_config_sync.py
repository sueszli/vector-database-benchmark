from google.cloud import websecurityscanner_v1alpha

def sample_get_scan_config():
    if False:
        print('Hello World!')
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    request = websecurityscanner_v1alpha.GetScanConfigRequest(name='name_value')
    response = client.get_scan_config(request=request)
    print(response)
from google.cloud import websecurityscanner_v1alpha

def sample_get_scan_run():
    if False:
        return 10
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    request = websecurityscanner_v1alpha.GetScanRunRequest(name='name_value')
    response = client.get_scan_run(request=request)
    print(response)
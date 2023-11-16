from google.cloud import websecurityscanner_v1alpha

def sample_start_scan_run():
    if False:
        i = 10
        return i + 15
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    request = websecurityscanner_v1alpha.StartScanRunRequest(name='name_value')
    response = client.start_scan_run(request=request)
    print(response)
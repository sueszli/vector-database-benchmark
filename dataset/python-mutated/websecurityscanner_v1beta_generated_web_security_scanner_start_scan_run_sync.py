from google.cloud import websecurityscanner_v1beta

def sample_start_scan_run():
    if False:
        return 10
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.StartScanRunRequest(name='name_value')
    response = client.start_scan_run(request=request)
    print(response)
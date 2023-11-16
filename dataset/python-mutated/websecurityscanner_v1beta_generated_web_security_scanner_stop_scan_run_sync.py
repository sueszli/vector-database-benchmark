from google.cloud import websecurityscanner_v1beta

def sample_stop_scan_run():
    if False:
        while True:
            i = 10
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.StopScanRunRequest(name='name_value')
    response = client.stop_scan_run(request=request)
    print(response)
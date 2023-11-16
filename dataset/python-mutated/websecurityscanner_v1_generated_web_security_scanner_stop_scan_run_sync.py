from google.cloud import websecurityscanner_v1

def sample_stop_scan_run():
    if False:
        for i in range(10):
            print('nop')
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.StopScanRunRequest()
    response = client.stop_scan_run(request=request)
    print(response)
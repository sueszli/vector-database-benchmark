from google.cloud import websecurityscanner_v1

def sample_get_scan_run():
    if False:
        for i in range(10):
            print('nop')
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.GetScanRunRequest()
    response = client.get_scan_run(request=request)
    print(response)
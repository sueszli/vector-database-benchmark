from google.cloud import websecurityscanner_v1

def sample_start_scan_run():
    if False:
        i = 10
        return i + 15
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.StartScanRunRequest()
    response = client.start_scan_run(request=request)
    print(response)
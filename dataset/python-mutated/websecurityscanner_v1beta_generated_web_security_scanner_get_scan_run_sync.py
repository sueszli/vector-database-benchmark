from google.cloud import websecurityscanner_v1beta

def sample_get_scan_run():
    if False:
        while True:
            i = 10
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.GetScanRunRequest(name='name_value')
    response = client.get_scan_run(request=request)
    print(response)
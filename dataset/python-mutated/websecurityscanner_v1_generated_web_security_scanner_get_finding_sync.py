from google.cloud import websecurityscanner_v1

def sample_get_finding():
    if False:
        i = 10
        return i + 15
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.GetFindingRequest()
    response = client.get_finding(request=request)
    print(response)
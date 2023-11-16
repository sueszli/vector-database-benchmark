from google.cloud import websecurityscanner_v1beta

def sample_get_finding():
    if False:
        for i in range(10):
            print('nop')
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.GetFindingRequest(name='name_value')
    response = client.get_finding(request=request)
    print(response)
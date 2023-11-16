from google.cloud import websecurityscanner_v1alpha

def sample_get_finding():
    if False:
        return 10
    client = websecurityscanner_v1alpha.WebSecurityScannerClient()
    request = websecurityscanner_v1alpha.GetFindingRequest(name='name_value')
    response = client.get_finding(request=request)
    print(response)
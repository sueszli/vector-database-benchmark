from google.cloud import websecurityscanner_v1

def sample_list_findings():
    if False:
        return 10
    client = websecurityscanner_v1.WebSecurityScannerClient()
    request = websecurityscanner_v1.ListFindingsRequest()
    page_result = client.list_findings(request=request)
    for response in page_result:
        print(response)
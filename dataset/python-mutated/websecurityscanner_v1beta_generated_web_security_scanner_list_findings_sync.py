from google.cloud import websecurityscanner_v1beta

def sample_list_findings():
    if False:
        i = 10
        return i + 15
    client = websecurityscanner_v1beta.WebSecurityScannerClient()
    request = websecurityscanner_v1beta.ListFindingsRequest(parent='parent_value', filter='filter_value')
    page_result = client.list_findings(request=request)
    for response in page_result:
        print(response)
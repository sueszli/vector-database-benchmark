from google.cloud import support_v2

def sample_list_cases():
    if False:
        for i in range(10):
            print('nop')
    client = support_v2.CaseServiceClient()
    request = support_v2.ListCasesRequest(parent='parent_value')
    page_result = client.list_cases(request=request)
    for response in page_result:
        print(response)
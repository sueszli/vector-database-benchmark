from google.cloud import support_v2

def sample_search_cases():
    if False:
        for i in range(10):
            print('nop')
    client = support_v2.CaseServiceClient()
    request = support_v2.SearchCasesRequest()
    page_result = client.search_cases(request=request)
    for response in page_result:
        print(response)
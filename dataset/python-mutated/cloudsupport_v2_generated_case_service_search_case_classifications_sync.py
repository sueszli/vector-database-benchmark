from google.cloud import support_v2

def sample_search_case_classifications():
    if False:
        i = 10
        return i + 15
    client = support_v2.CaseServiceClient()
    request = support_v2.SearchCaseClassificationsRequest()
    page_result = client.search_case_classifications(request=request)
    for response in page_result:
        print(response)
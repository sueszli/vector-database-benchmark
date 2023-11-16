from google.cloud import talent_v4

def sample_list_companies():
    if False:
        return 10
    client = talent_v4.CompanyServiceClient()
    request = talent_v4.ListCompaniesRequest(parent='parent_value')
    page_result = client.list_companies(request=request)
    for response in page_result:
        print(response)
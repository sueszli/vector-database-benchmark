from google.cloud import talent_v4beta1

def sample_list_companies():
    if False:
        for i in range(10):
            print('nop')
    client = talent_v4beta1.CompanyServiceClient()
    request = talent_v4beta1.ListCompaniesRequest(parent='parent_value')
    page_result = client.list_companies(request=request)
    for response in page_result:
        print(response)
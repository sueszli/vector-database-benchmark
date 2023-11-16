from google.cloud import talent_v4beta1

def sample_get_company():
    if False:
        print('Hello World!')
    client = talent_v4beta1.CompanyServiceClient()
    request = talent_v4beta1.GetCompanyRequest(name='name_value')
    response = client.get_company(request=request)
    print(response)
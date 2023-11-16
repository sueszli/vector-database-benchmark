from google.cloud import talent_v4

def sample_get_company():
    if False:
        while True:
            i = 10
    client = talent_v4.CompanyServiceClient()
    request = talent_v4.GetCompanyRequest(name='name_value')
    response = client.get_company(request=request)
    print(response)
from google.cloud import talent_v4

def sample_delete_company():
    if False:
        while True:
            i = 10
    client = talent_v4.CompanyServiceClient()
    request = talent_v4.DeleteCompanyRequest(name='name_value')
    client.delete_company(request=request)
from google.cloud import talent_v4beta1

def sample_delete_company():
    if False:
        for i in range(10):
            print('nop')
    client = talent_v4beta1.CompanyServiceClient()
    request = talent_v4beta1.DeleteCompanyRequest(name='name_value')
    client.delete_company(request=request)
from google.cloud import talent_v4beta1

def sample_update_company():
    if False:
        while True:
            i = 10
    client = talent_v4beta1.CompanyServiceClient()
    company = talent_v4beta1.Company()
    company.display_name = 'display_name_value'
    company.external_id = 'external_id_value'
    request = talent_v4beta1.UpdateCompanyRequest(company=company)
    response = client.update_company(request=request)
    print(response)
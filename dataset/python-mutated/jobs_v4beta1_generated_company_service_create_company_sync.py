from google.cloud import talent_v4beta1

def sample_create_company():
    if False:
        i = 10
        return i + 15
    client = talent_v4beta1.CompanyServiceClient()
    company = talent_v4beta1.Company()
    company.display_name = 'display_name_value'
    company.external_id = 'external_id_value'
    request = talent_v4beta1.CreateCompanyRequest(parent='parent_value', company=company)
    response = client.create_company(request=request)
    print(response)
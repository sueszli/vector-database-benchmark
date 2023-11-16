from google.cloud import talent

def get_company(project_id, tenant_id, company_id):
    if False:
        for i in range(10):
            print('nop')
    'Get Company'
    client = talent.CompanyServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(company_id, bytes):
        company_id = company_id.decode('utf-8')
    name = client.company_path(project_id, tenant_id, company_id)
    response = client.get_company(name=name)
    print(f'Company name: {response.name}')
    print(f'Display name: {response.display_name}')
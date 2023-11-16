from google.cloud import talent

def delete_company(project_id, tenant_id, company_id):
    if False:
        i = 10
        return i + 15
    'Delete Company'
    client = talent.CompanyServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(company_id, bytes):
        company_id = company_id.decode('utf-8')
    name = client.company_path(project_id, tenant_id, company_id)
    client.delete_company(name=name)
    print('Deleted company')
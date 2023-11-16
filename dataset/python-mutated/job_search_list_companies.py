from google.cloud import talent

def list_companies(project_id, tenant_id):
    if False:
        i = 10
        return i + 15
    'List Companies'
    client = talent.CompanyServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    parent = f'projects/{project_id}/tenants/{tenant_id}'
    results = []
    for company in client.list_companies(parent=parent):
        results.append(company.name)
        print(f'Company Name: {company.name}')
        print(f'Display Name: {company.display_name}')
        print(f'External ID: {company.external_id}')
    return results
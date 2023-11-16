from google.cloud import talent

def create_company(project_id, tenant_id, display_name, external_id):
    if False:
        print('Hello World!')
    'Create Company'
    client = talent.CompanyServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(display_name, bytes):
        display_name = display_name.decode('utf-8')
    if isinstance(external_id, bytes):
        external_id = external_id.decode('utf-8')
    parent = f'projects/{project_id}/tenants/{tenant_id}'
    company = {'display_name': display_name, 'external_id': external_id}
    response = client.create_company(parent=parent, company=company)
    print('Created Company')
    print(f'Name: {response.name}')
    print(f'Display Name: {response.display_name}')
    print(f'External ID: {response.external_id}')
    return response.name
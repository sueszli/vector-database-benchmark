from google.cloud import talent

def get_tenant(project_id, tenant_id):
    if False:
        i = 10
        return i + 15
    'Get Tenant by name'
    client = talent.TenantServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    name = client.tenant_path(project_id, tenant_id)
    response = client.get_tenant(name=name)
    print(f'Name: {response.name}')
    print(f'External ID: {response.external_id}')
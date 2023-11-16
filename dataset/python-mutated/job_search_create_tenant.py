from google.cloud import talent

def create_tenant(project_id, external_id):
    if False:
        print('Hello World!')
    'Create Tenant for scoping resources, e.g. companies and jobs'
    client = talent.TenantServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(external_id, bytes):
        external_id = external_id.decode('utf-8')
    parent = f'projects/{project_id}'
    tenant = talent.Tenant(external_id=external_id)
    response = client.create_tenant(parent=parent, tenant=tenant)
    print('Created Tenant')
    print(f'Name: {response.name}')
    print(f'External ID: {response.external_id}')
    return response.name
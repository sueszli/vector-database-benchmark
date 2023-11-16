from google.cloud import talent

def list_tenants(project_id):
    if False:
        return 10
    'List Tenants'
    client = talent.TenantServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    parent = f'projects/{project_id}'
    for response_item in client.list_tenants(parent=parent):
        print(f'Tenant Name: {response_item.name}')
        print(f'External ID: {response_item.external_id}')
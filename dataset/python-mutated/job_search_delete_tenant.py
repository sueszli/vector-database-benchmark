from google.cloud import talent

def delete_tenant(project_id, tenant_id):
    if False:
        print('Hello World!')
    'Delete Tenant'
    client = talent.TenantServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    name = client.tenant_path(project_id, tenant_id)
    client.delete_tenant(name=name)
    print('Deleted Tenant.')
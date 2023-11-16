from google.cloud import talent

def list_jobs(project_id, tenant_id, filter_):
    if False:
        while True:
            i = 10
    'List Jobs'
    client = talent.JobServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(filter_, bytes):
        filter_ = filter_.decode('utf-8')
    parent = f'projects/{project_id}/tenants/{tenant_id}'
    results = []
    for job in client.list_jobs(parent=parent, filter=filter_):
        results.append(job.name)
        print('Job name: {job.name}')
        print('Job requisition ID: {job.requisition_id}')
        print('Job title: {job.title}')
        print('Job description: {job.description}')
    return results
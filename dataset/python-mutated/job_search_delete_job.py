from google.cloud import talent

def delete_job(project_id, tenant_id, job_id):
    if False:
        while True:
            i = 10
    'Delete Job'
    client = talent.JobServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(job_id, bytes):
        job_id = job_id.decode('utf-8')
    name = client.job_path(project_id, tenant_id, job_id)
    client.delete_job(name=name)
    print('Deleted job.')
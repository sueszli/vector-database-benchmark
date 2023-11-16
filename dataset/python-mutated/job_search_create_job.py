from google.cloud import talent

def create_job(project_id, tenant_id, company_id, requisition_id, job_application_url):
    if False:
        while True:
            i = 10
    'Create Job'
    client = talent.JobServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(company_id, bytes):
        company_id = company_id.decode('utf-8')
    if isinstance(requisition_id, bytes):
        requisition_id = requisition_id.decode('utf-8')
    if isinstance(job_application_url, bytes):
        job_application_url = job_application_url.decode('utf-8')
    parent = f'projects/{project_id}/tenants/{tenant_id}'
    uris = [job_application_url]
    application_info = {'uris': uris}
    addresses = ['1600 Amphitheatre Parkway, Mountain View, CA 94043', '111 8th Avenue, New York, NY 10011']
    job = {'company': company_id, 'requisition_id': requisition_id, 'title': 'Software Developer', 'description': 'Develop, maintain the software solutions.', 'application_info': application_info, 'addresses': addresses, 'language_code': 'en-US'}
    response = client.create_job(parent=parent, job=job)
    print(f'Created job: {response.name}')
    return response.name
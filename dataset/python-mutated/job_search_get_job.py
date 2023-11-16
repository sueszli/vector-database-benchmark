from google.cloud import talent

def get_job(project_id, tenant_id, job_id):
    if False:
        print('Hello World!')
    'Get Job'
    client = talent.JobServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(job_id, bytes):
        job_id = job_id.decode('utf-8')
    name = client.job_path(project_id, tenant_id, job_id)
    response = client.get_job(name=name)
    print(f'Job name: {response.name}')
    print(f'Requisition ID: {response.requisition_id}')
    print(f'Title: {response.title}')
    print(f'Description: {response.description}')
    print(f'Posting language: {response.language_code}')
    for address in response.addresses:
        print(f'Address: {address}')
    for email in response.application_info.emails:
        print(f'Email: {email}')
    for website_uri in response.application_info.uris:
        print(f'Website: {website_uri}')
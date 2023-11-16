from google.cloud import talent

def create_job(project_id, tenant_id, company_id, requisition_id):
    if False:
        return 10
    'Create Job with Custom Attributes'
    client = talent.JobServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(company_id, bytes):
        company_id = company_id.decode('utf-8')
    custom_attribute = talent.CustomAttribute()
    custom_attribute.filterable = True
    custom_attribute.string_values.append('Intern')
    custom_attribute.string_values.append('Apprenticeship')
    parent = f'projects/{project_id}/tenants/{tenant_id}'
    job = talent.Job(company=company_id, title='Software Engineer', requisition_id=requisition_id, description='This is a description of this job', language_code='en-us', custom_attributes={'FOR_STUDENTS': custom_attribute})
    response = client.create_job(parent=parent, job=job)
    print(f'Created job: {response.name}')
    return response.name
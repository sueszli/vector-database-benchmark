from google.cloud import talent

def search_jobs(project_id, tenant_id):
    if False:
        i = 10
        return i + 15
    'Search Jobs using custom rankings'
    client = talent.JobServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    parent = f'projects/{project_id}/tenants/{tenant_id}'
    domain = 'www.example.com'
    session_id = 'Hashed session identifier'
    user_id = 'Hashed user identifier'
    request_metadata = talent.RequestMetadata(domain=domain, session_id=session_id, user_id=user_id)
    importance_level = talent.SearchJobsRequest.CustomRankingInfo.ImportanceLevel.EXTREME
    ranking_expression = '(someFieldLong + 25) * 0.25'
    custom_ranking_info = {'importance_level': importance_level, 'ranking_expression': ranking_expression}
    order_by = 'custom_ranking desc'
    results = []
    request = talent.SearchJobsRequest(parent=parent, request_metadata=request_metadata, custom_ranking_info=custom_ranking_info, order_by=order_by)
    for response_item in client.search_jobs(request=request).matching_jobs:
        print(f'Job summary: {response_item.job_summary}')
        print(f'Job title snippet: {response_item.job_title_snippet}')
        job = response_item.job
        results.append(job.name)
        print(f'Job name: {job.name}')
        print(f'Job title: {job.title}')
    return results
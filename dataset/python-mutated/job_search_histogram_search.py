from google.cloud import talent

def search_jobs(project_id, tenant_id, query):
    if False:
        return 10
    '\n    Search Jobs with histogram queries\n\n    Args:\n      query Histogram query\n      More info on histogram facets, constants, and built-in functions:\n      https://godoc.org/google.golang.org/genproto/googleapis/cloud/talent/v4beta1#SearchJobsRequest\n    '
    client = talent.JobServiceClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(query, bytes):
        query = query.decode('utf-8')
    parent = f'projects/{project_id}/tenants/{tenant_id}'
    domain = 'www.example.com'
    session_id = 'Hashed session identifier'
    user_id = 'Hashed user identifier'
    request_metadata = {'domain': domain, 'session_id': session_id, 'user_id': user_id}
    histogram_queries_element = {'histogram_query': query}
    histogram_queries = [histogram_queries_element]
    results = []
    request = talent.SearchJobsRequest(parent=parent, request_metadata=request_metadata, histogram_queries=histogram_queries)
    for response_item in client.search_jobs(request=request).matching_jobs:
        print('Job summary: {response_item.job_summary}')
        print('Job title snippet: {response_item.job_title_snippet}')
        job = response_item.job
        results.append(job)
        print('Job name: {job.name}')
        print('Job title: {job.title}')
    return results
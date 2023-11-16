from google.cloud import talent_v4beta1

def complete_query(project_id, tenant_id, query):
    if False:
        for i in range(10):
            print('nop')
    'Complete job title given partial text (autocomplete)'
    client = talent_v4beta1.CompletionClient()
    if isinstance(project_id, bytes):
        project_id = project_id.decode('utf-8')
    if isinstance(tenant_id, bytes):
        tenant_id = tenant_id.decode('utf-8')
    if isinstance(query, bytes):
        query = query.decode('utf-8')
    parent = f'projects/{project_id}/tenants/{tenant_id}'
    request = talent_v4beta1.CompleteQueryRequest(parent=parent, query=query, page_size=5, language_codes=['en-US'])
    response = client.complete_query(request=request)
    for result in response.completion_results:
        print(f'Suggested title: {result.suggestion}')
        print(f'Suggestion type: {talent_v4beta1.CompleteQueryRequest.CompletionType(result.type_).name}')
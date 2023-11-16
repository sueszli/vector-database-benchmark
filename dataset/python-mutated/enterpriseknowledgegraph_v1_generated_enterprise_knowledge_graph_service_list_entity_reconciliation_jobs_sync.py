from google.cloud import enterpriseknowledgegraph_v1

def sample_list_entity_reconciliation_jobs():
    if False:
        print('Hello World!')
    client = enterpriseknowledgegraph_v1.EnterpriseKnowledgeGraphServiceClient()
    request = enterpriseknowledgegraph_v1.ListEntityReconciliationJobsRequest(parent='parent_value')
    page_result = client.list_entity_reconciliation_jobs(request=request)
    for response in page_result:
        print(response)
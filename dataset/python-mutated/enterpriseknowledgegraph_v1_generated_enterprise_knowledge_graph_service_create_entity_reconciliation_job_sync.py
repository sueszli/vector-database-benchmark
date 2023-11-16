from google.cloud import enterpriseknowledgegraph_v1

def sample_create_entity_reconciliation_job():
    if False:
        i = 10
        return i + 15
    client = enterpriseknowledgegraph_v1.EnterpriseKnowledgeGraphServiceClient()
    request = enterpriseknowledgegraph_v1.CreateEntityReconciliationJobRequest(parent='parent_value')
    response = client.create_entity_reconciliation_job(request=request)
    print(response)
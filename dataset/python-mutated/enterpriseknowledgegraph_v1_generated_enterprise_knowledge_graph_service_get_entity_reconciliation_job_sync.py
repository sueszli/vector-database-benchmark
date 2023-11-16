from google.cloud import enterpriseknowledgegraph_v1

def sample_get_entity_reconciliation_job():
    if False:
        while True:
            i = 10
    client = enterpriseknowledgegraph_v1.EnterpriseKnowledgeGraphServiceClient()
    request = enterpriseknowledgegraph_v1.GetEntityReconciliationJobRequest(name='name_value')
    response = client.get_entity_reconciliation_job(request=request)
    print(response)
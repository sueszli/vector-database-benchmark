from google.cloud import enterpriseknowledgegraph_v1

def sample_delete_entity_reconciliation_job():
    if False:
        i = 10
        return i + 15
    client = enterpriseknowledgegraph_v1.EnterpriseKnowledgeGraphServiceClient()
    request = enterpriseknowledgegraph_v1.DeleteEntityReconciliationJobRequest(name='name_value')
    client.delete_entity_reconciliation_job(request=request)
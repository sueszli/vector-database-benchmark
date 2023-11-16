from google.cloud import enterpriseknowledgegraph_v1

def sample_cancel_entity_reconciliation_job():
    if False:
        print('Hello World!')
    client = enterpriseknowledgegraph_v1.EnterpriseKnowledgeGraphServiceClient()
    request = enterpriseknowledgegraph_v1.CancelEntityReconciliationJobRequest(name='name_value')
    client.cancel_entity_reconciliation_job(request=request)
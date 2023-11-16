from google.cloud import enterpriseknowledgegraph as ekg

def cancel_entity_reconciliation_job_sample(project_id: str, location: str, job_id: str) -> None:
    if False:
        print('Hello World!')
    client = ekg.EnterpriseKnowledgeGraphServiceClient()
    name = client.entity_reconciliation_job_path(project=project_id, location=location, entity_reconciliation_job=job_id)
    request = ekg.CancelEntityReconciliationJobRequest(name=name)
    client.cancel_entity_reconciliation_job(request=request)
    print(f'Job: {name} successfully cancelled')
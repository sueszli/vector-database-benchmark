from google.cloud import enterpriseknowledgegraph as ekg

def delete_entity_reconciliation_job_sample(project_id: str, location: str, job_id: str) -> None:
    if False:
        while True:
            i = 10
    client = ekg.EnterpriseKnowledgeGraphServiceClient()
    name = client.entity_reconciliation_job_path(project=project_id, location=location, entity_reconciliation_job=job_id)
    request = ekg.DeleteEntityReconciliationJobRequest(name=name)
    client.delete_entity_reconciliation_job(request=request)
    print(f'Job: {name} successfully deleted')
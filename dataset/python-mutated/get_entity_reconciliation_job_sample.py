from google.cloud import enterpriseknowledgegraph as ekg

def get_entity_reconciliation_job_sample(project_id: str, location: str, job_id: str) -> None:
    if False:
        i = 10
        return i + 15
    client = ekg.EnterpriseKnowledgeGraphServiceClient()
    name = client.entity_reconciliation_job_path(project=project_id, location=location, entity_reconciliation_job=job_id)
    request = ekg.GetEntityReconciliationJobRequest(name=name)
    response = client.get_entity_reconciliation_job(request=request)
    print(f'Job: {response.name}')
    print(f'Input Table: {response.input_config.bigquery_input_configs[0].bigquery_table}')
    print(f'Output Dataset: {response.output_config.bigquery_dataset}')
    print(f'State: {response.state.name}')
from google.cloud import enterpriseknowledgegraph as ekg

def list_entity_reconciliation_jobs_sample(project_id: str, location: str) -> None:
    if False:
        i = 10
        return i + 15
    client = ekg.EnterpriseKnowledgeGraphServiceClient()
    parent = client.common_location_path(project=project_id, location=location)
    request = ekg.ListEntityReconciliationJobsRequest(parent=parent)
    pager = client.list_entity_reconciliation_jobs(request=request)
    for response in pager:
        print(f'Job: {response.name}')
        print(f'Input Table: {response.input_config.bigquery_input_configs[0].bigquery_table}')
        print(f'Output Dataset: {response.output_config.bigquery_dataset}')
        print(f'State: {response.state.name}\n')
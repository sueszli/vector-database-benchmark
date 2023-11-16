from google.cloud import enterpriseknowledgegraph as ekg

def create_entity_reconciliation_job_sample(project_id: str, location: str, input_dataset: str, input_table: str, mapping_file_uri: str, entity_type: int, output_dataset: str) -> None:
    if False:
        print('Hello World!')
    client = ekg.EnterpriseKnowledgeGraphServiceClient()
    parent = client.common_location_path(project=project_id, location=location)
    input_config = ekg.InputConfig(bigquery_input_configs=[ekg.BigQueryInputConfig(bigquery_table=client.table_path(project=project_id, dataset=input_dataset, table=input_table), gcs_uri=mapping_file_uri)], entity_type=entity_type)
    output_config = ekg.OutputConfig(bigquery_dataset=client.dataset_path(project=project_id, dataset=output_dataset))
    entity_reconciliation_job = ekg.EntityReconciliationJob(input_config=input_config, output_config=output_config)
    request = ekg.CreateEntityReconciliationJobRequest(parent=parent, entity_reconciliation_job=entity_reconciliation_job)
    response = client.create_entity_reconciliation_job(request=request)
    print(f'Job: {response.name}')
    print(f'Input Table: {response.input_config.bigquery_input_configs[0].bigquery_table}')
    print(f'Output Dataset: {response.output_config.bigquery_dataset}')
    print(f'State: {response.state.name}')
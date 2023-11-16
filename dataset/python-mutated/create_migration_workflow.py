def create_migration_workflow(gcs_input_path: str, gcs_output_path: str, project_id: str) -> None:
    if False:
        return 10
    'Creates a migration workflow of a Batch SQL Translation and prints the response.'
    from google.cloud import bigquery_migration_v2
    parent = f'projects/{project_id}/locations/us'
    client = bigquery_migration_v2.MigrationServiceClient()
    source_dialect = bigquery_migration_v2.Dialect()
    source_dialect.teradata_dialect = bigquery_migration_v2.TeradataDialect(mode=bigquery_migration_v2.TeradataDialect.Mode.SQL)
    target_dialect = bigquery_migration_v2.Dialect()
    target_dialect.bigquery_dialect = bigquery_migration_v2.BigQueryDialect()
    translation_config = bigquery_migration_v2.TranslationConfigDetails(gcs_source_path=gcs_input_path, gcs_target_path=gcs_output_path, source_dialect=source_dialect, target_dialect=target_dialect)
    migration_task = bigquery_migration_v2.MigrationTask(type_='Translation_Teradata2BQ', translation_config_details=translation_config)
    workflow = bigquery_migration_v2.MigrationWorkflow(display_name='demo-workflow-python-example-Teradata2BQ')
    workflow.tasks['translation-task'] = migration_task
    request = bigquery_migration_v2.CreateMigrationWorkflowRequest(parent=parent, migration_workflow=workflow)
    response = client.create_migration_workflow(request=request)
    print('Created workflow:')
    print(response.display_name)
    print('Current state:')
    print(response.State(response.state))
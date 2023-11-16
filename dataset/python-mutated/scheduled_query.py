def create_scheduled_query(override_values={}):
    if False:
        while True:
            i = 10
    from google.cloud import bigquery_datatransfer
    transfer_client = bigquery_datatransfer.DataTransferServiceClient()
    project_id = 'your-project-id'
    dataset_id = 'your_dataset_id'
    service_account_name = 'abcdef-test-sa@abcdef-test.iam.gserviceaccount.com'
    project_id = override_values.get('project_id', project_id)
    dataset_id = override_values.get('dataset_id', dataset_id)
    service_account_name = override_values.get('service_account_name', service_account_name)
    query_string = '\n    SELECT\n      CURRENT_TIMESTAMP() as current_time,\n      @run_time as intended_run_time,\n      @run_date as intended_run_date,\n      17 as some_integer\n    '
    parent = transfer_client.common_project_path(project_id)
    transfer_config = bigquery_datatransfer.TransferConfig(destination_dataset_id=dataset_id, display_name='Your Scheduled Query Name', data_source_id='scheduled_query', params={'query': query_string, 'destination_table_name_template': 'your_table_{run_date}', 'write_disposition': 'WRITE_TRUNCATE', 'partitioning_field': ''}, schedule='every 24 hours')
    transfer_config = transfer_client.create_transfer_config(bigquery_datatransfer.CreateTransferConfigRequest(parent=parent, transfer_config=transfer_config, service_account_name=service_account_name))
    print("Created scheduled query '{}'".format(transfer_config.name))
    return transfer_config
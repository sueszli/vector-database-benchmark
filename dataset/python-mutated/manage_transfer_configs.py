def list_configs(override_values={}):
    if False:
        return 10
    from google.cloud import bigquery_datatransfer
    transfer_client = bigquery_datatransfer.DataTransferServiceClient()
    project_id = 'my-project'
    project_id = override_values.get('project_id', project_id)
    parent = transfer_client.common_project_path(project_id)
    configs = transfer_client.list_transfer_configs(parent=parent)
    print('Got the following configs:')
    for config in configs:
        print(f'\tID: {config.name}, Schedule: {config.schedule}')

def update_config(override_values={}):
    if False:
        print('Hello World!')
    from google.cloud import bigquery_datatransfer
    from google.protobuf import field_mask_pb2
    transfer_client = bigquery_datatransfer.DataTransferServiceClient()
    transfer_config_name = 'projects/1234/locations/us/transferConfigs/abcd'
    new_display_name = 'My Transfer Config'
    new_display_name = override_values.get('new_display_name', new_display_name)
    transfer_config_name = override_values.get('transfer_config_name', transfer_config_name)
    transfer_config = bigquery_datatransfer.TransferConfig(name=transfer_config_name)
    transfer_config.display_name = new_display_name
    transfer_config = transfer_client.update_transfer_config({'transfer_config': transfer_config, 'update_mask': field_mask_pb2.FieldMask(paths=['display_name'])})
    print(f"Updated config: '{transfer_config.name}'")
    print(f"New display name: '{transfer_config.display_name}'")
    return transfer_config

def update_credentials_with_service_account(override_values={}):
    if False:
        i = 10
        return i + 15
    from google.cloud import bigquery_datatransfer
    from google.protobuf import field_mask_pb2
    transfer_client = bigquery_datatransfer.DataTransferServiceClient()
    service_account_name = 'abcdef-test-sa@abcdef-test.iam.gserviceaccount.com'
    transfer_config_name = 'projects/1234/locations/us/transferConfigs/abcd'
    service_account_name = override_values.get('service_account_name', service_account_name)
    transfer_config_name = override_values.get('transfer_config_name', transfer_config_name)
    transfer_config = bigquery_datatransfer.TransferConfig(name=transfer_config_name)
    transfer_config = transfer_client.update_transfer_config({'transfer_config': transfer_config, 'update_mask': field_mask_pb2.FieldMask(paths=['service_account_name']), 'service_account_name': service_account_name})
    print("Updated config: '{}'".format(transfer_config.name))
    return transfer_config

def schedule_backfill_manual_transfer(override_values={}):
    if False:
        while True:
            i = 10
    import datetime
    from google.cloud.bigquery_datatransfer_v1 import DataTransferServiceClient, StartManualTransferRunsRequest
    client = DataTransferServiceClient()
    transfer_config_name = 'projects/1234/locations/us/transferConfigs/abcd'
    transfer_config_name = override_values.get('transfer_config_name', transfer_config_name)
    now = datetime.datetime.now(datetime.timezone.utc)
    start_time = now - datetime.timedelta(days=5)
    end_time = now - datetime.timedelta(days=2)
    start_time = datetime.datetime(start_time.year, start_time.month, start_time.day, tzinfo=datetime.timezone.utc)
    end_time = datetime.datetime(end_time.year, end_time.month, end_time.day, tzinfo=datetime.timezone.utc)
    requested_time_range = StartManualTransferRunsRequest.TimeRange(start_time=start_time, end_time=end_time)
    request = StartManualTransferRunsRequest(parent=transfer_config_name, requested_time_range=requested_time_range)
    response = client.start_manual_transfer_runs(request=request)
    print('Started manual transfer runs:')
    for run in response.runs:
        print(f'backfill: {run.run_time} run: {run.name}')
    return response.runs

def delete_config(override_values={}):
    if False:
        while True:
            i = 10
    import google.api_core.exceptions
    from google.cloud import bigquery_datatransfer
    transfer_client = bigquery_datatransfer.DataTransferServiceClient()
    transfer_config_name = 'projects/1234/locations/us/transferConfigs/abcd'
    transfer_config_name = override_values.get('transfer_config_name', transfer_config_name)
    try:
        transfer_client.delete_transfer_config(name=transfer_config_name)
    except google.api_core.exceptions.NotFound:
        print('Transfer config not found.')
    else:
        print(f'Deleted transfer config: {transfer_config_name}')
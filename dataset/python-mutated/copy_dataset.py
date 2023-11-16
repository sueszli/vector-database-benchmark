def copy_dataset(override_values={}):
    if False:
        i = 10
        return i + 15
    from google.cloud import bigquery_datatransfer
    transfer_client = bigquery_datatransfer.DataTransferServiceClient()
    destination_project_id = 'my-destination-project'
    destination_dataset_id = 'my_destination_dataset'
    source_project_id = 'my-source-project'
    source_dataset_id = 'my_source_dataset'
    destination_project_id = override_values.get('destination_project_id', destination_project_id)
    destination_dataset_id = override_values.get('destination_dataset_id', destination_dataset_id)
    source_project_id = override_values.get('source_project_id', source_project_id)
    source_dataset_id = override_values.get('source_dataset_id', source_dataset_id)
    transfer_config = bigquery_datatransfer.TransferConfig(destination_dataset_id=destination_dataset_id, display_name='Your Dataset Copy Name', data_source_id='cross_region_copy', params={'source_project_id': source_project_id, 'source_dataset_id': source_dataset_id}, schedule='every 24 hours')
    transfer_config = transfer_client.create_transfer_config(parent=transfer_client.common_project_path(destination_project_id), transfer_config=transfer_config)
    print(f'Created transfer config: {transfer_config.name}')
    return transfer_config
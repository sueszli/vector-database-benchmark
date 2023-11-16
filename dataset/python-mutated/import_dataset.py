def import_dataset(project_id, dataset_id, path):
    if False:
        for i in range(10):
            print('nop')
    'Import a dataset.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    dataset_full_id = client.dataset_path(project_id, 'us-central1', dataset_id)
    input_uris = path.split(',')
    gcs_source = automl.GcsSource(input_uris=input_uris)
    input_config = automl.InputConfig(gcs_source=gcs_source)
    response = client.import_data(name=dataset_full_id, input_config=input_config)
    print('Processing import...')
    print(f'Data imported. {response.result()}')
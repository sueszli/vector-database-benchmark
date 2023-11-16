def export_dataset(project_id, dataset_id, gcs_uri):
    if False:
        for i in range(10):
            print('nop')
    'Export a dataset.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    dataset_full_id = client.dataset_path(project_id, 'us-central1', dataset_id)
    gcs_destination = automl.GcsDestination(output_uri_prefix=gcs_uri)
    output_config = automl.OutputConfig(gcs_destination=gcs_destination)
    response = client.export_data(name=dataset_full_id, output_config=output_config)
    print(f'Dataset exported. {response.result()}')
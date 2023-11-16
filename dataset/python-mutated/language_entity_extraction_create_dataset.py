def create_dataset(project_id, display_name):
    if False:
        while True:
            i = 10
    'Create a dataset.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    project_location = f'projects/{project_id}/locations/us-central1'
    metadata = automl.TextExtractionDatasetMetadata()
    dataset = automl.Dataset(display_name=display_name, text_extraction_dataset_metadata=metadata)
    response = client.create_dataset(parent=project_location, dataset=dataset)
    created_dataset = response.result()
    print(f'Dataset name: {created_dataset.name}')
    print('Dataset id: {}'.format(created_dataset.name.split('/')[-1]))
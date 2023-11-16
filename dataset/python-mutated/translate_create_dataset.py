def create_dataset(project_id, display_name):
    if False:
        i = 10
        return i + 15
    'Create a dataset.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    project_location = f'projects/{project_id}/locations/us-central1'
    dataset_metadata = automl.TranslationDatasetMetadata(source_language_code='en', target_language_code='ja')
    dataset = automl.Dataset(display_name=display_name, translation_dataset_metadata=dataset_metadata)
    response = client.create_dataset(parent=project_location, dataset=dataset)
    created_dataset = response.result()
    print(f'Dataset name: {created_dataset.name}')
    print('Dataset id: {}'.format(created_dataset.name.split('/')[-1]))
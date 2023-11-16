def create_dataset(project_id, display_name):
    if False:
        for i in range(10):
            print('nop')
    'Create a dataset.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    project_location = f'projects/{project_id}/locations/us-central1'
    metadata = automl.ImageClassificationDatasetMetadata(classification_type=automl.ClassificationType.MULTILABEL)
    dataset = automl.Dataset(display_name=display_name, image_classification_dataset_metadata=metadata)
    response = client.create_dataset(parent=project_location, dataset=dataset, timeout=300)
    created_dataset = response.result()
    print(f'Dataset name: {created_dataset.name}')
    print('Dataset id: {}'.format(created_dataset.name.split('/')[-1]))
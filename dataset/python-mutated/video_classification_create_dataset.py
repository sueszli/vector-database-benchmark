from google.cloud import automl_v1beta1 as automl

def create_dataset(project_id='YOUR_PROJECT_ID', display_name='your_datasets_display_name'):
    if False:
        for i in range(10):
            print('nop')
    'Create a automl video classification dataset.'
    client = automl.AutoMlClient()
    project_location = f'projects/{project_id}/locations/us-central1'
    metadata = automl.VideoClassificationDatasetMetadata()
    dataset = automl.Dataset(display_name=display_name, video_classification_dataset_metadata=metadata)
    created_dataset = client.create_dataset(parent=project_location, dataset=dataset)
    print(f'Dataset name: {created_dataset.name}')
    print('Dataset id: {}'.format(created_dataset.name.split('/')[-1]))
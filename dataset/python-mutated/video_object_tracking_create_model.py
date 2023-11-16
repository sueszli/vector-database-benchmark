from google.cloud import automl_v1beta1 as automl

def create_model(project_id='YOUR_PROJECT_ID', dataset_id='YOUR_DATASET_ID', display_name='your_models_display_name'):
    if False:
        return 10
    'Create a automl video classification model.'
    client = automl.AutoMlClient()
    project_location = f'projects/{project_id}/locations/us-central1'
    metadata = automl.VideoObjectTrackingModelMetadata()
    model = automl.Model(display_name=display_name, dataset_id=dataset_id, video_object_tracking_model_metadata=metadata)
    response = client.create_model(parent=project_location, model=model)
    print(f'Training operation name: {response.operation.name}')
    print('Training started...')
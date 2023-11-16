def create_model(project_id, dataset_id, display_name):
    if False:
        print('Hello World!')
    'Create a model.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    project_location = f'projects/{project_id}/locations/us-central1'
    metadata = automl.ImageObjectDetectionModelMetadata(train_budget_milli_node_hours=24000)
    model = automl.Model(display_name=display_name, dataset_id=dataset_id, image_object_detection_model_metadata=metadata)
    response = client.create_model(parent=project_location, model=model)
    print(f'Training operation name: {response.operation.name}')
    print('Training started...')
    return response
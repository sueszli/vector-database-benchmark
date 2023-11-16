from google.cloud import automl_v1

def sample_create_dataset():
    if False:
        return 10
    client = automl_v1.AutoMlClient()
    dataset = automl_v1.Dataset()
    dataset.translation_dataset_metadata.source_language_code = 'source_language_code_value'
    dataset.translation_dataset_metadata.target_language_code = 'target_language_code_value'
    request = automl_v1.CreateDatasetRequest(parent='parent_value', dataset=dataset)
    operation = client.create_dataset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
from google.cloud import automl_v1beta1

def sample_create_dataset():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1beta1.AutoMlClient()
    dataset = automl_v1beta1.Dataset()
    dataset.translation_dataset_metadata.source_language_code = 'source_language_code_value'
    dataset.translation_dataset_metadata.target_language_code = 'target_language_code_value'
    request = automl_v1beta1.CreateDatasetRequest(parent='parent_value', dataset=dataset)
    response = client.create_dataset(request=request)
    print(response)
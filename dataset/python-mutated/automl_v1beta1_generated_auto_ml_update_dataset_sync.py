from google.cloud import automl_v1beta1

def sample_update_dataset():
    if False:
        i = 10
        return i + 15
    client = automl_v1beta1.AutoMlClient()
    dataset = automl_v1beta1.Dataset()
    dataset.translation_dataset_metadata.source_language_code = 'source_language_code_value'
    dataset.translation_dataset_metadata.target_language_code = 'target_language_code_value'
    request = automl_v1beta1.UpdateDatasetRequest(dataset=dataset)
    response = client.update_dataset(request=request)
    print(response)
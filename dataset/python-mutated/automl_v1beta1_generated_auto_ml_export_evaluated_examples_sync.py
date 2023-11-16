from google.cloud import automl_v1beta1

def sample_export_evaluated_examples():
    if False:
        i = 10
        return i + 15
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.ExportEvaluatedExamplesRequest(name='name_value')
    operation = client.export_evaluated_examples(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
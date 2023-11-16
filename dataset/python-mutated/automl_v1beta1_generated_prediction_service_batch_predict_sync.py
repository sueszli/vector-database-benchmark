from google.cloud import automl_v1beta1

def sample_batch_predict():
    if False:
        while True:
            i = 10
    client = automl_v1beta1.PredictionServiceClient()
    request = automl_v1beta1.BatchPredictRequest(name='name_value')
    operation = client.batch_predict(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
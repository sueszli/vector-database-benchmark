from google.cloud import automl_v1

def sample_predict():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1.PredictionServiceClient()
    payload = automl_v1.ExamplePayload()
    payload.image.image_bytes = b'image_bytes_blob'
    request = automl_v1.PredictRequest(name='name_value', payload=payload)
    response = client.predict(request=request)
    print(response)
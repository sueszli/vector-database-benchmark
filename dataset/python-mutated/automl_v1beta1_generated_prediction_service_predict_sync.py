from google.cloud import automl_v1beta1

def sample_predict():
    if False:
        while True:
            i = 10
    client = automl_v1beta1.PredictionServiceClient()
    payload = automl_v1beta1.ExamplePayload()
    payload.image.image_bytes = b'image_bytes_blob'
    request = automl_v1beta1.PredictRequest(name='name_value', payload=payload)
    response = client.predict(request=request)
    print(response)
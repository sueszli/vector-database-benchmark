from google.cloud import retail_v2beta

def sample_tune_model():
    if False:
        return 10
    client = retail_v2beta.ModelServiceClient()
    request = retail_v2beta.TuneModelRequest(name='name_value')
    operation = client.tune_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
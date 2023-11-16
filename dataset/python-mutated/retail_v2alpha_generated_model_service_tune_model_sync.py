from google.cloud import retail_v2alpha

def sample_tune_model():
    if False:
        while True:
            i = 10
    client = retail_v2alpha.ModelServiceClient()
    request = retail_v2alpha.TuneModelRequest(name='name_value')
    operation = client.tune_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
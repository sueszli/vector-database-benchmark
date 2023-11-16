from google.cloud import retail_v2

def sample_tune_model():
    if False:
        print('Hello World!')
    client = retail_v2.ModelServiceClient()
    request = retail_v2.TuneModelRequest(name='name_value')
    operation = client.tune_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
from google.cloud import retail_v2beta

def sample_create_model():
    if False:
        while True:
            i = 10
    client = retail_v2beta.ModelServiceClient()
    model = retail_v2beta.Model()
    model.name = 'name_value'
    model.display_name = 'display_name_value'
    model.type_ = 'type__value'
    request = retail_v2beta.CreateModelRequest(parent='parent_value', model=model)
    operation = client.create_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
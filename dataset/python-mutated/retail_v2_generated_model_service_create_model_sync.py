from google.cloud import retail_v2

def sample_create_model():
    if False:
        print('Hello World!')
    client = retail_v2.ModelServiceClient()
    model = retail_v2.Model()
    model.name = 'name_value'
    model.display_name = 'display_name_value'
    model.type_ = 'type__value'
    request = retail_v2.CreateModelRequest(parent='parent_value', model=model)
    operation = client.create_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
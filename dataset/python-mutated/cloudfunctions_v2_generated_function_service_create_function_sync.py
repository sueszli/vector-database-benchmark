from google.cloud import functions_v2

def sample_create_function():
    if False:
        while True:
            i = 10
    client = functions_v2.FunctionServiceClient()
    request = functions_v2.CreateFunctionRequest(parent='parent_value')
    operation = client.create_function(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
from google.cloud import functions_v2

def sample_update_function():
    if False:
        print('Hello World!')
    client = functions_v2.FunctionServiceClient()
    request = functions_v2.UpdateFunctionRequest()
    operation = client.update_function(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
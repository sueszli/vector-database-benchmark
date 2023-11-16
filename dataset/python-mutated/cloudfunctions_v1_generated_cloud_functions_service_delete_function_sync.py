from google.cloud import functions_v1

def sample_delete_function():
    if False:
        print('Hello World!')
    client = functions_v1.CloudFunctionsServiceClient()
    request = functions_v1.DeleteFunctionRequest(name='name_value')
    operation = client.delete_function(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
from google.cloud import functions_v1

def sample_get_function():
    if False:
        for i in range(10):
            print('nop')
    client = functions_v1.CloudFunctionsServiceClient()
    request = functions_v1.GetFunctionRequest(name='name_value')
    response = client.get_function(request=request)
    print(response)
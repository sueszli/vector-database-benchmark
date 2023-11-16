from google.cloud import functions_v1

def sample_call_function():
    if False:
        i = 10
        return i + 15
    client = functions_v1.CloudFunctionsServiceClient()
    request = functions_v1.CallFunctionRequest(name='name_value', data='data_value')
    response = client.call_function(request=request)
    print(response)
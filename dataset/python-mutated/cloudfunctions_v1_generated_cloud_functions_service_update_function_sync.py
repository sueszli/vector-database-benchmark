from google.cloud import functions_v1

def sample_update_function():
    if False:
        print('Hello World!')
    client = functions_v1.CloudFunctionsServiceClient()
    function = functions_v1.CloudFunction()
    function.source_archive_url = 'source_archive_url_value'
    request = functions_v1.UpdateFunctionRequest(function=function)
    operation = client.update_function(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
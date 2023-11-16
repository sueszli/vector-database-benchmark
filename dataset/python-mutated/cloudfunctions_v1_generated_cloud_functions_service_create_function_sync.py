from google.cloud import functions_v1

def sample_create_function():
    if False:
        print('Hello World!')
    client = functions_v1.CloudFunctionsServiceClient()
    function = functions_v1.CloudFunction()
    function.source_archive_url = 'source_archive_url_value'
    request = functions_v1.CreateFunctionRequest(location='location_value', function=function)
    operation = client.create_function(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
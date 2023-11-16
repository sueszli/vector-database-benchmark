from google.cloud import functions_v1

def sample_list_functions():
    if False:
        return 10
    client = functions_v1.CloudFunctionsServiceClient()
    request = functions_v1.ListFunctionsRequest()
    page_result = client.list_functions(request=request)
    for response in page_result:
        print(response)
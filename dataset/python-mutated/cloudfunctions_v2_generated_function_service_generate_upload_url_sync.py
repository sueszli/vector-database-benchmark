from google.cloud import functions_v2

def sample_generate_upload_url():
    if False:
        return 10
    client = functions_v2.FunctionServiceClient()
    request = functions_v2.GenerateUploadUrlRequest(parent='parent_value')
    response = client.generate_upload_url(request=request)
    print(response)
from google.cloud import functions_v1

def sample_generate_upload_url():
    if False:
        i = 10
        return i + 15
    client = functions_v1.CloudFunctionsServiceClient()
    request = functions_v1.GenerateUploadUrlRequest()
    response = client.generate_upload_url(request=request)
    print(response)
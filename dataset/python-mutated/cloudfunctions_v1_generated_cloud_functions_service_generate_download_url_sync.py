from google.cloud import functions_v1

def sample_generate_download_url():
    if False:
        while True:
            i = 10
    client = functions_v1.CloudFunctionsServiceClient()
    request = functions_v1.GenerateDownloadUrlRequest()
    response = client.generate_download_url(request=request)
    print(response)
from google.cloud import functions_v2

def sample_generate_download_url():
    if False:
        while True:
            i = 10
    client = functions_v2.FunctionServiceClient()
    request = functions_v2.GenerateDownloadUrlRequest(name='name_value')
    response = client.generate_download_url(request=request)
    print(response)
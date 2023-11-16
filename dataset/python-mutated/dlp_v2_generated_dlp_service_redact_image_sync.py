from google.cloud import dlp_v2

def sample_redact_image():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.RedactImageRequest()
    response = client.redact_image(request=request)
    print(response)
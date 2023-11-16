from google.cloud import dlp_v2

def sample_deidentify_content():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.DeidentifyContentRequest()
    response = client.deidentify_content(request=request)
    print(response)
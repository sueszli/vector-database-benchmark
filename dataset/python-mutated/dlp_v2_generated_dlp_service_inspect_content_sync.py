from google.cloud import dlp_v2

def sample_inspect_content():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.InspectContentRequest()
    response = client.inspect_content(request=request)
    print(response)
from google.cloud import dlp_v2

def sample_reidentify_content():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.ReidentifyContentRequest(parent='parent_value')
    response = client.reidentify_content(request=request)
    print(response)
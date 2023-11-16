from google.cloud import dlp_v2

def sample_list_info_types():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.ListInfoTypesRequest()
    response = client.list_info_types(request=request)
    print(response)
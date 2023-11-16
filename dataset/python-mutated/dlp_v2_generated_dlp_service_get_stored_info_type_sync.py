from google.cloud import dlp_v2

def sample_get_stored_info_type():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.GetStoredInfoTypeRequest(name='name_value')
    response = client.get_stored_info_type(request=request)
    print(response)
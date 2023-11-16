from google.cloud import dlp_v2

def sample_update_stored_info_type():
    if False:
        i = 10
        return i + 15
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.UpdateStoredInfoTypeRequest(name='name_value')
    response = client.update_stored_info_type(request=request)
    print(response)
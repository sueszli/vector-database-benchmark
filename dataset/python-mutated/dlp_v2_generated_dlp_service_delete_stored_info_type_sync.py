from google.cloud import dlp_v2

def sample_delete_stored_info_type():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.DeleteStoredInfoTypeRequest(name='name_value')
    client.delete_stored_info_type(request=request)
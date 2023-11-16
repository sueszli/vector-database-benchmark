from google.cloud import dlp_v2

def sample_create_stored_info_type():
    if False:
        for i in range(10):
            print('nop')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.CreateStoredInfoTypeRequest(parent='parent_value')
    response = client.create_stored_info_type(request=request)
    print(response)
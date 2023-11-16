from google.cloud import dlp_v2

def sample_list_stored_info_types():
    if False:
        while True:
            i = 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.ListStoredInfoTypesRequest(parent='parent_value')
    page_result = client.list_stored_info_types(request=request)
    for response in page_result:
        print(response)
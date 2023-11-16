from google.cloud import securitycenter_v1

def sample_list_mute_configs():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.ListMuteConfigsRequest(parent='parent_value')
    page_result = client.list_mute_configs(request=request)
    for response in page_result:
        print(response)
from google.cloud import securitycenter_v1

def sample_get_mute_config():
    if False:
        i = 10
        return i + 15
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.GetMuteConfigRequest(name='name_value')
    response = client.get_mute_config(request=request)
    print(response)
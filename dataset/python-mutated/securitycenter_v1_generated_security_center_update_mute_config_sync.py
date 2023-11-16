from google.cloud import securitycenter_v1

def sample_update_mute_config():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1.SecurityCenterClient()
    mute_config = securitycenter_v1.MuteConfig()
    mute_config.filter = 'filter_value'
    request = securitycenter_v1.UpdateMuteConfigRequest(mute_config=mute_config)
    response = client.update_mute_config(request=request)
    print(response)
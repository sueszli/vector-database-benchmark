from google.cloud import securitycenter_v1

def sample_create_mute_config():
    if False:
        while True:
            i = 10
    client = securitycenter_v1.SecurityCenterClient()
    mute_config = securitycenter_v1.MuteConfig()
    mute_config.filter = 'filter_value'
    request = securitycenter_v1.CreateMuteConfigRequest(parent='parent_value', mute_config=mute_config, mute_config_id='mute_config_id_value')
    response = client.create_mute_config(request=request)
    print(response)
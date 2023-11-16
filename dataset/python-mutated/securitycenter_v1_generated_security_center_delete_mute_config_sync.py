from google.cloud import securitycenter_v1

def sample_delete_mute_config():
    if False:
        print('Hello World!')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.DeleteMuteConfigRequest(name='name_value')
    client.delete_mute_config(request=request)
from google.cloud import securitycenter_v1

def sample_set_mute():
    if False:
        while True:
            i = 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.SetMuteRequest(name='name_value', mute='UNDEFINED')
    response = client.set_mute(request=request)
    print(response)
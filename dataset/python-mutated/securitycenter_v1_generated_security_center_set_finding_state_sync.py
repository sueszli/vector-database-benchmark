from google.cloud import securitycenter_v1

def sample_set_finding_state():
    if False:
        i = 10
        return i + 15
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.SetFindingStateRequest(name='name_value', state='INACTIVE')
    response = client.set_finding_state(request=request)
    print(response)
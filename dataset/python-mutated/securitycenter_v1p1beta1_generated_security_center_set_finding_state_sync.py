from google.cloud import securitycenter_v1p1beta1

def sample_set_finding_state():
    if False:
        while True:
            i = 10
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.SetFindingStateRequest(name='name_value', state='INACTIVE')
    response = client.set_finding_state(request=request)
    print(response)
from google.cloud import securitycenter_v1beta1

def sample_update_finding():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = securitycenter_v1beta1.UpdateFindingRequest()
    response = client.update_finding(request=request)
    print(response)
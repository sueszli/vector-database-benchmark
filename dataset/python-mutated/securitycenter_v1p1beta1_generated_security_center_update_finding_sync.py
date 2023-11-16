from google.cloud import securitycenter_v1p1beta1

def sample_update_finding():
    if False:
        i = 10
        return i + 15
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.UpdateFindingRequest()
    response = client.update_finding(request=request)
    print(response)
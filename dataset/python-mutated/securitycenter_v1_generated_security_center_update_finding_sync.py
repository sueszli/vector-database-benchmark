from google.cloud import securitycenter_v1

def sample_update_finding():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.UpdateFindingRequest()
    response = client.update_finding(request=request)
    print(response)
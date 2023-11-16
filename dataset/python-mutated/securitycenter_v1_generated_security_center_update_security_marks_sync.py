from google.cloud import securitycenter_v1

def sample_update_security_marks():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.UpdateSecurityMarksRequest()
    response = client.update_security_marks(request=request)
    print(response)
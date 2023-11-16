from google.cloud import securitycenter_v1p1beta1

def sample_update_security_marks():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.UpdateSecurityMarksRequest()
    response = client.update_security_marks(request=request)
    print(response)
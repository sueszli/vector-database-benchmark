from google.cloud import securitycenter_v1beta1

def sample_update_source():
    if False:
        i = 10
        return i + 15
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = securitycenter_v1beta1.UpdateSourceRequest()
    response = client.update_source(request=request)
    print(response)
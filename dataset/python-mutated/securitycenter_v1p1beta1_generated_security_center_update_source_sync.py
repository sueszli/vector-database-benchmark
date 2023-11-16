from google.cloud import securitycenter_v1p1beta1

def sample_update_source():
    if False:
        print('Hello World!')
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.UpdateSourceRequest()
    response = client.update_source(request=request)
    print(response)
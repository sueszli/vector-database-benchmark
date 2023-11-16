from google.cloud import securitycenter_v1p1beta1

def sample_get_source():
    if False:
        while True:
            i = 10
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = securitycenter_v1p1beta1.GetSourceRequest(name='name_value')
    response = client.get_source(request=request)
    print(response)
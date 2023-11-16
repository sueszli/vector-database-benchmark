from google.cloud import securitycenter_v1

def sample_get_source():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.GetSourceRequest(name='name_value')
    response = client.get_source(request=request)
    print(response)
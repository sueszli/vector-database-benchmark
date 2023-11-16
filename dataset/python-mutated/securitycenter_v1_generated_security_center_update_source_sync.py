from google.cloud import securitycenter_v1

def sample_update_source():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.UpdateSourceRequest()
    response = client.update_source(request=request)
    print(response)
from google.cloud import securitycenter_v1

def sample_update_external_system():
    if False:
        while True:
            i = 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.UpdateExternalSystemRequest()
    response = client.update_external_system(request=request)
    print(response)
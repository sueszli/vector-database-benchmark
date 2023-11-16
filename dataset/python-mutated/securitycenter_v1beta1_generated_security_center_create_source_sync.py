from google.cloud import securitycenter_v1beta1

def sample_create_source():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = securitycenter_v1beta1.CreateSourceRequest(parent='parent_value')
    response = client.create_source(request=request)
    print(response)
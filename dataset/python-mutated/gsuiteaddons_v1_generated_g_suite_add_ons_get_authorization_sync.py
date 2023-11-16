from google.cloud import gsuiteaddons_v1

def sample_get_authorization():
    if False:
        for i in range(10):
            print('nop')
    client = gsuiteaddons_v1.GSuiteAddOnsClient()
    request = gsuiteaddons_v1.GetAuthorizationRequest(name='name_value')
    response = client.get_authorization(request=request)
    print(response)
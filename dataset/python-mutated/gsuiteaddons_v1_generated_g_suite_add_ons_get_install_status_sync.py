from google.cloud import gsuiteaddons_v1

def sample_get_install_status():
    if False:
        print('Hello World!')
    client = gsuiteaddons_v1.GSuiteAddOnsClient()
    request = gsuiteaddons_v1.GetInstallStatusRequest(name='name_value')
    response = client.get_install_status(request=request)
    print(response)
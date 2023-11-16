from google.cloud import appengine_admin_v1

def sample_get_version():
    if False:
        while True:
            i = 10
    client = appengine_admin_v1.VersionsClient()
    request = appengine_admin_v1.GetVersionRequest()
    response = client.get_version(request=request)
    print(response)
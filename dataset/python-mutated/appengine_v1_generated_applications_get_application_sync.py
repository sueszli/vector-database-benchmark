from google.cloud import appengine_admin_v1

def sample_get_application():
    if False:
        i = 10
        return i + 15
    client = appengine_admin_v1.ApplicationsClient()
    request = appengine_admin_v1.GetApplicationRequest()
    response = client.get_application(request=request)
    print(response)
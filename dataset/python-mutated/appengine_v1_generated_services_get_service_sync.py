from google.cloud import appengine_admin_v1

def sample_get_service():
    if False:
        for i in range(10):
            print('nop')
    client = appengine_admin_v1.ServicesClient()
    request = appengine_admin_v1.GetServiceRequest()
    response = client.get_service(request=request)
    print(response)
from google.cloud import appengine_admin_v1

def sample_list_services():
    if False:
        while True:
            i = 10
    client = appengine_admin_v1.ServicesClient()
    request = appengine_admin_v1.ListServicesRequest()
    page_result = client.list_services(request=request)
    for response in page_result:
        print(response)
from google.cloud import appengine_admin_v1

def sample_list_versions():
    if False:
        while True:
            i = 10
    client = appengine_admin_v1.VersionsClient()
    request = appengine_admin_v1.ListVersionsRequest()
    page_result = client.list_versions(request=request)
    for response in page_result:
        print(response)
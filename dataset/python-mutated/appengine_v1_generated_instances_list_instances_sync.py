from google.cloud import appengine_admin_v1

def sample_list_instances():
    if False:
        for i in range(10):
            print('nop')
    client = appengine_admin_v1.InstancesClient()
    request = appengine_admin_v1.ListInstancesRequest()
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)
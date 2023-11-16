from google.cloud import appengine_admin_v1

def sample_get_instance():
    if False:
        while True:
            i = 10
    client = appengine_admin_v1.InstancesClient()
    request = appengine_admin_v1.GetInstanceRequest()
    response = client.get_instance(request=request)
    print(response)
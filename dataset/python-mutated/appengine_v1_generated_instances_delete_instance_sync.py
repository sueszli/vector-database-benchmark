from google.cloud import appengine_admin_v1

def sample_delete_instance():
    if False:
        return 10
    client = appengine_admin_v1.InstancesClient()
    request = appengine_admin_v1.DeleteInstanceRequest()
    operation = client.delete_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)
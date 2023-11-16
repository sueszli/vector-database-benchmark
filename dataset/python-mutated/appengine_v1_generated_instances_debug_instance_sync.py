from google.cloud import appengine_admin_v1

def sample_debug_instance():
    if False:
        print('Hello World!')
    client = appengine_admin_v1.InstancesClient()
    request = appengine_admin_v1.DebugInstanceRequest()
    operation = client.debug_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)